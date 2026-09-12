
  auto n = thread_position_in_grid.z;
  auto b_idx = n / Hv;
  auto hv_idx = n % Hv;
  auto hk_idx = hv_idx / (Hv / Hk);

  auto dv_idx = thread_position_in_grid.y * 16;
  const short sg_id = thread_position_in_threadgroup.y; // 0..3

  const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
  const short qid = simd_lane_id >> 2;
  const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));

  // set up pointers
  // g: [B, T, Hv]
  auto g_ = g + b_idx * T * Hv;

  // q, k: [B, T, Hk, Dk]
  auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
  auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

  // v, y: [B, T, Hv, Dv]
  y += b_idx * T * Hv * Dv + hv_idx * Dv;
  auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
  auto beta_ = beta + b_idx * T * Hv;

  // state_in, state_out: [B, Hv, Dv, Dk]
  auto i_state = state_in + (n * Dv + dv_idx) * Dk;
  auto o_state = state_out + (n * Dv + dv_idx) * Dk;

  threadgroup float gamma_all[C * 4];
  threadgroup float* gamma = gamma_all + sg_id * C;

  float beta_fm[2];

  mlx::steel::NAXTile<float, 1, Dk / 16> S_tile;
  S_tile.load(i_state, Dk);

  mlx::steel::NAXTile<float, 1, 2> K_tile, Q_tile;
  mlx::steel::NAXTile<float, 1, Dk / 16> W_tile; // panel

  mlx::steel::NAXTile<float, 1, 1> V_tile;
  mlx::steel::NAXTile<float, 1, 1> U_tile;
  mlx::steel::NAXTile<float, 1, 1> WS_tile;
  mlx::steel::NAXTile<float, 1, 1> delta_tile;
  mlx::steel::NAXTile<float, 1, 1> tmp_tile;
  mlx::steel::NAXTile<float, 1, 1> QKt_tile;
  mlx::steel::NAXTile<float, 1, 1> out_tile;
  mlx::steel::NAXTile<float, 1, 1> Tinv_tile, P;

  mlx::steel::NAXTile<float, 1, 1> KKtK_tile, KKt_tile;

  mlx::steel::NAXTile<float, 1, 1> I_tile;
  STEEL_PRAGMA_UNROLL
  for (short _i = 0; _i < decltype(I_tile)::kElemsPerFrag; _i++) {
    const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); /* {fn, fm} */
    const short _fn = _c.x;
    const short _fm = _c.y;
    AT_NAX(I_tile, _i) = (_fn == _fm) ? 1.0f : 0.0f;
  }
  mlx::steel::NAXTile<float, 1, 1> TMP_tile;

  auto process_chunk = [&](const short valid_rows,
                           auto bounded_tag) __attribute__((always_inline)) {
    constexpr bool B = decltype(bounded_tag)::value;

    auto load_seq = [&](thread auto& tile, auto src, int ld) {
      if constexpr (B) {
        tile.load_rows(src, ld, valid_rows);
      } else {
        tile.load(src, ld);
      }
    };

    float g_val = (thread_index_in_simdgroup < (uint)valid_rows)
        ? metal::fast::log(
              metal::max(
                  static_cast<float>(
                      g_[thread_index_in_simdgroup * Hv + hv_idx]),
                  1e-6f))
        : 0.0f;

    auto gamma_val = simd_prefix_inclusive_sum(g_val);
    if (thread_index_in_simdgroup < C) {
      gamma[thread_index_in_simdgroup] = static_cast<float>(gamma_val);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    beta_fm[0] = (fm < valid_rows) ? beta_[fm * Hv + hv_idx] : 0.0f;
    const short fm1 = fm + mlx::steel::BaseNAXFrag::kElemRowsJump;
    beta_fm[1] = (fm1 < valid_rows) ? beta_[fm1 * Hv + hv_idx] : 0.0f;

    KKt_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Dk * Hk);
      MMA16x16x32(KKt_tile, 0, K_tile, false, 0, K_tile, true, 0, 0);
    }

    KKtK_tile = KKt_tile;

    SCALE_TRIEQ_NAX1(KKtK_tile, beta_fm);
    SUB_NAX(Tinv_tile, I_tile, KKtK_tile);
    STEEL_PRAGMA_UNROLL
    for (int step = 0; step < 15; step++) {
      MM16x16x16(TMP_tile, 0, KKtK_tile, false, 0, Tinv_tile, false, 0, 1);
      SUB_NAX(Tinv_tile, I_tile, TMP_tile);
    }

    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < Dk / 16; nn += 2) {
      load_seq(K_tile, k_ + nn * 16, Dk * Hk);
      SCALE_BETA_NAX(K_tile, beta_fm);
      MM16x32x16(W_tile, nn, Tinv_tile, false, 0, K_tile, false, 0, 2);
    }
    SCALE_ROW_NAX(W_tile, gamma)

    SCALE_TRI_NAX(Tinv_tile, gamma)
    load_seq(V_tile, v_ + dv_idx, Dv * Hv);
    SCALE_BETA_NAX(V_tile, beta_fm);
    MM16x16x16(U_tile, 0, Tinv_tile, false, 0, V_tile, false, 0, 3)

        WS_tile.clear();
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < Dk / 16; kk += 2) {
      MMA16x16x32(WS_tile, 0, W_tile, false, kk, S_tile, true, kk, 4)
    }

    SUB_NAX(delta_tile, U_tile, WS_tile)

    tmp_tile.clear();
    QKt_tile.clear();
    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(Q_tile, q_ + kk, Hk * Dk);
      load_seq(K_tile, k_ + kk, Hk * Dk);

      MMA16x16x32(QKt_tile, 0, Q_tile, false, 0, K_tile, true, 0, 5);

      SCALE_ROW_NAX(Q_tile, gamma);
      MMA16x16x32(tmp_tile, 0, Q_tile, false, 0, S_tile, true, kk / 16, 6);
    }

    SCALE_TRI_NAX(QKt_tile, gamma)

    out_tile = tmp_tile;
    MMA16x16x16(out_tile, 0, QKt_tile, false, 0, delta_tile, false, 0, 7);

    STEEL_PRAGMA_UNROLL
    for (short _i = 0; _i < decltype(out_tile)::kElemsPerFrag; _i++) {
      const short2 _c = mlx::steel::BaseNAXFrag::get_coord(_i); // {fn, fm}
      const short _fn = _c.x;
      const short _fm = _c.y;
      if (_fm < valid_rows) {
        y[_fm * Hv * Dv + dv_idx + _fn] =
            static_cast<InT>(AT_NAX(out_tile, _i));
      }
    }

    SCALE_NAX(S_tile, metal::fast::exp(gamma[C - 1]));

    for (int kk = 0; kk < Dk; kk += 32) {
      load_seq(K_tile, k_ + kk, Hk * Dk);
      SCALE2_NAX(K_tile, gamma);
      MMA16x32x16(S_tile, kk / 16, delta_tile, true, 0, K_tile, false, 0, 8);
    }
  };

  int t = 0;
  for (; t + C <= T; t += C) {
    process_chunk(C, metal::false_type{});
    q_ += C * Hk * Dk;
    k_ += C * Hk * Dk;
    v_ += C * Hv * Dv;
    beta_ += C * Hv;
    y += C * Hv * Dv;
    g_ += C * Hv;
  }
  if (t < T) {
    process_chunk(short(T - t), metal::true_type{});
  }

  S_tile.store(o_state, Dk);
