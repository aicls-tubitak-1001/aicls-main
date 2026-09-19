# Test Stratejisi

Karar: [ADR-0010](../adr/0010-test-stratejisi.md). İlke: test-önce; deterministik çekirdek yüksek kapsamla, model kenarları sentetik fikstürlerle CPU'da; veri seti, ağırlık ve GPU gerektiren testler işaretlenir ve CI'da koşmaz.

## 1. Seviyeler ve araçlar

| Seviye | Araç | Kapsam | Süre hedefi |
|---|---|---|---|
| Birim | pytest 9, pytest-xdist, pytest-timeout | saf fonksiyonlar, I/O yok | < 100 ms/test |
| Özellik tabanlı | hypothesis | ayrıştırıcı gidiş-dönüşü, normalizasyon idempotansı, metrik değişmezleri, fuzz | PR: 50 örnek, gece: 1000 |
| Entegrasyon | httpx `ASGITransport` + FastAPI, SQLite `:memory:` (Alembic head), stub modeller | API, DB, WS, kitap yükleme, boru hattı | < 60 s/test |
| Uçtan uca | pytest (`e2e`) + Playwright (`browser`) | metin/iz/ses → KaTeX sonucu; chromium/firefox/webkit | gece / yerel |
| Performans | pytest (`perf`) | aşama bütçeleri, kitap çözümleme p50 | gece + rapor öncesi |
| Kapı | pytest (`gated`) | betiklerin onaysız reddi, `check_gates.py` | her PR |
| GPU / veri / MySQL | pytest (`gpu`, `data`, `mysql`) | gerçek CUDA, gerçek veri seti, gerçek MySQL | yalnızca iş istasyonu / gece |

Ön uç: vitest + @testing-library/react (birim), Playwright (e2e, üç tarayıcı = öneri A.1.2 Chrome/Firefox/Safari).

## 2. İşaretler (`pyproject [tool.pytest.ini_options] markers`)

`unit` (varsayılan), `property`, `integration`, `ws`, `e2e`, `perf`, `slow` (> 10 s), `gated`, `gpu` (`torch.cuda.is_available()` değilse atla), `data` (`AICLS_DATA_ROOT/<set>/manifest.json` doğrulanmazsa atla), `mysql` (`AICLS_TEST_MYSQL_URL` yoksa atla), `browser`.

PR seçimi: `-m "not gpu and not data and not mysql and not slow and not browser"`. Gece: `slow or perf or mysql or browser`.

## 3. Kapsam hedefleri (`scripts/coverage_gate.py`)

| Paket | Eşik |
|---|---|
| genel | 85 |
| `aicls.latex2sympy`, `aicls.mathengine`, `aicls.nlu` | 95 |
| `aicls.eval` (metrics.py, normalizers.py ≥ 98) | 95 |
| `aicls.core` | 90 |
| `aicls.book`, `aicls.collect` | 85 |
| `aicls.api`, `aicls.db` | 80 |
| `aicls.data`, `aicls.ml` | 75 |
| `aicls.ocr`, `aicls.asr` | 70 |
| `web/` (vitest) | 70 |

Kapsam düşürme yalnızca ADR ile.

## 4. Fikstür stratejisi (`tests/conftest.py`)

| Fikstür | İçerik |
|---|---|
| `settings` | `AICLS_ENV=test`, `db_url=sqlite:///:memory:`, `models_enabled=False`, `data_root=tmp_path`, sabit `jwt_secret`, `contrib_enabled=True`, `ethics_approval_ref="TEST"` |
| `engine`, `db_session` | `StaticPool` SQLite; şema `alembic upgrade head` ile |
| `app`, `client` | `create_app(settings)`; bağımlılık geçersiz kılmalarıyla `StubRecognizer`, `StubTranscriber`, `InProcessExecutor` |
| `pool_client` | `PoolExecutor` ile aynı uygulama (sandbox testleri için) |
| `user_token(role)` | JWT üretici fabrika |
| `engine_cas` | `MathEngine(InProcessExecutor(), timeout_s=2.0)` |
| `inventory`, `resolver` | envanter + `parse_command` bağlaması |
| `rng` | `numpy.random.default_rng(1234)` |
| `synthetic_ink`, `synthetic_inkml_dir` | Bezier izleri, bilinen LaTeX etiketleri; `synth_inkml_dataset(n=64)` |
| `synthetic_wave` | sinüs + gürültü, 1–3 s, 16 kHz, bilinen SNR/kırpılma |
| `synthetic_wav_bytes` | PCM16 WAV baytı (RIFF başlıklı) |
| `mini_book` | `datasets/book/sample/` gerçek yükleyiciyle yüklenmiş (≥ 10 birim, 12 üçlü) |
| `tiny_ocr_cfg`, `tiny_asr_cfg` | `training/configs/*/smoke_synthetic.yaml` |
| `curriculum_rows`, `valid_inputs`, `nlu_cases` | JSONL yükleyici; `id` alanı pytest kimliği |
| `freeze_time` | freezegun |

Sentetik fikstürler `scripts/make_synthetic_fixtures.py` ile üretilir (seed 1234), `tests/fixtures/synthetic/` altında git'te (< 2 MB): 8 InkML, 8 PNG, 8 WAV, 2 manifest. Görüntüler matplotlib mathtext ile üretilir (TeX kurulumu gerekmez).

## 5. Özellik tabanlı testler (`tests/property/`)

`tests/property/strategies.py::st_expr` — `Num | Sym | Add | Mul | Div | Pow | Root | Func | Integral | Derivative | Limit` üzerinde özyinelemeli (derinlik ≤ 5), `st_turkish_text`, `st_latex_noise`.

- `test_property_roundtrip_ast_to_latex_to_ast` — `parse(to_latex(ast)) == ast`
- `test_property_latex_to_sympy_symbolically_equivalent_after_reprint` — `simplify(A − B) == 0`
- `test_property_parse_never_raises_non_aicls_error` — rastgele Unicode
- `test_property_normalize_latex_idempotent`, `test_property_latex_v1_idempotent`, `test_property_tr_v1_idempotent`, `test_property_nlu_normalizer_idempotent`
- `test_property_cer_zero_iff_identical`, `test_property_cer_symmetric_bounds`, `test_property_wer_bounded_by_one_plus_insertions`, `test_property_wer_invariant_under_punctuation`
- `test_property_parse_command_never_raises_non_aicls_error`
- `test_property_number_words_roundtrip_0_to_9999`

## 6. Somut test adları

### latex2sympy
`test_parse_frac_nested_returns_div_node`, `test_parse_implicit_multiplication_2x_sin_x`, `test_parse_derivative_leibniz_frac_d_dx`, `test_parse_definite_integral_bounds_to_sympy_integral`, `test_parse_indefinite_integral_dx_detected`, `test_parse_limit_with_direction_plus`, `test_parse_greek_alpha_maps_to_symbol_alpha`, `test_parse_sin_x_y_binds_first_power`, `test_parse_sum_with_limits`, `test_parse_unsupported_cases_raises_with_construct_name`, `test_parse_error_reports_line_and_column`, `test_unbalanced_braces_is_parse_error_not_crash`, `test_normalize_latex_records_single_brace_repair`, `test_sympy_one_third_prints_frac_1_3`, `test_sympy_latex_output_is_within_grammar_on_curriculum`, `test_differential_against_sympy_parse_latex_subset`, `test_curriculum_and_valid_inputs_parse_success_rate_above_0_99`, `test_frozen_curriculum_rows_unchanged`

### mathengine
`test_diff_x_squared_equals_2x`, `test_integrate_x_squared_0_1_equals_one_third`, `test_solve_x_squared_minus_4_returns_minus2_and_2`, `test_limit_sin_x_over_x_at_0_is_1`, `test_execute_timeout_kills_worker_and_pool_recovers`, `test_pool_executor_prewarms_sympy_import`, `test_execute_timeout_returns_cas_timeout_within_budget`, `test_unsupported_problem_returns_contract_turkish_message`, `test_cache_hit_on_repeated_command_under_5ms`, `test_plot_renders_deterministic_png_hash`, `test_symbolic_equivalent_sin2_plus_cos2_vs_1`, `test_input_over_2048_chars_rejected`, `test_curriculum_problems_cas_correct_rate_above_0_99` (`slow`)

### nlu
Bkz. [metin-komut-katmani.md §8](../mimari/metin-komut-katmani.md#8-değerlendirme-ve-testler) — 21 test adı.

### ocr / asr
`test_inkml_read_write_roundtrip_synthetic`, `test_ink_from_canvas_json_preserves_timestamps`, `test_inkml_parser_rejects_external_entities`, `test_render_ink_shape_and_range_0_1`, `test_preprocess_steps_are_ablatable_and_ordered`, `test_preprocess_noop_when_all_flags_off`, `test_tokenizer_encode_decode_roundtrip_on_curriculum_latex`, `test_stub_recognizer_deterministic`, `test_seq2seq_forward_shapes_tiny`, `test_beam_search_returns_k_sorted`, `test_torchvision_encoders_built_with_weights_none`, `test_read_wav_rejects_png_bytes_with_wav_extension`, `test_read_wav_accepts_44100_and_resamples_to_16k`, `test_audio_qc_flags_clipping_above_threshold`, `test_audio_qc_flags_low_snr`, `test_audio_qc_rejects_duration_out_of_range`, `test_audio_qc_flags_silence_ratio`, `test_log_mel_shape_80_mels`, `test_augment_pitch_shift_preserves_duration`, `test_spec_augment_masks_time_and_freq`, `test_stub_transcriber_deterministic`, `test_whisper_ft_builds_from_tiny_random_config_without_download`

### book / collect
`test_markdown_tagger_recognises_tanim_1_1`, `test_markdown_tagger_recognises_block_syntax_with_aliases`, `test_coverage_is_one_for_sample_chapter`, `test_loader_rejects_untagged_heading_with_list`, `test_generate_aliases_turkish_suffixes_turev`, `test_book_resolve_turevin_tanimi_and_turev_tanimi_same_unit`, `test_book_search_like_fallback_on_sqlite`, `test_book_fetch_test_set_accuracy_above_0_95`, `test_book_export_turtle_reparses_with_rdflib_triple_count_equals_edges`, `test_jsonld_roundtrip_unit`, `test_consent_required_before_recording`, `test_recording_rejected_by_qc_writes_no_file`, `test_next_prompts_prefers_least_recorded`, `test_withdraw_deletes_files_keeps_consent_audit`, `test_export_manifest_has_no_identity_fields`, `test_export_manifest_excludes_withdrawn_and_qc_failed`

### data / eval / ml
Bkz. [egitim-altyapisi.md §10](../ml/egitim-altyapisi.md#10-testlerin-altyapıyı-nasıl-kullandığı) ve [degerlendirme.md §6](../ml/degerlendirme.md#6-testler).

### api (entegrasyon)
`test_api_register_login_refresh_flow`, `test_api_refresh_rotation_invalidates_old_cookie`, `test_api_login_lockout_after_10_failures`, `test_api_rbac_matrix_every_router_every_role`, `test_api_student_cannot_activate_model_403`, `test_api_problem_json_shape_on_parse_error_with_position`, `test_api_compute_text_end_to_end_stub_latency_recorded`, `test_api_compute_text_returns_clarification_200_not_error`, `test_api_compute_latex_diff_returns_2x`, `test_api_unsupported_returns_422_with_contract_message`, `test_api_server_timing_header_present`, `test_api_telemetry_latency_stores_client_and_server_ms`, `test_api_ocr_upload_rejects_non_png_magic`, `test_api_ocr_ink_json_over_20000_points_rejected`, `test_api_asr_transcribe_wav_only`, `test_api_contribute_requires_consent_before_upload`, `test_api_contribute_upload_runs_qc_and_returns_reasons_tr`, `test_api_contribute_blocked_without_ethics_ref_503`, `test_api_book_fetch_turev_tanimi_returns_definition_unit`, `test_api_feedback_sus_score_computed`, `test_api_sql_injection_string_in_search_is_parameterized`, `test_api_collect_profile_mounts_only_allowed_routers`, `test_api_history_export_tex_and_pdf`, `test_api_rate_limit_returns_429_with_retry_after`

### WebSocket
`test_ws_rejects_expired_ticket`, `test_ws_ink_stream_partial_then_final`, `test_ws_ink_slow_down_when_flooded`, `test_ws_audio_backpressure`, `test_ws_audio_final_then_command_with_stub`

### db
`test_alembic_upgrade_head_on_sqlite_and_mysql_dialect_compile`, `test_migrations_head_base_head_sqlite`, `test_autogenerate_diff_is_empty`, `test_schema_rules_all_strings_have_length`, `test_migrations_mysql_service` (`mysql`)

### gated
`test_gated_download_refuses_without_flag_and_env`, `test_gated_download_refuses_without_ledger_line`, `test_gated_train_refuses_in_ci`, `test_gated_train_allows_synthetic_without_gate`, `test_install_ml_script_contains_sm120_check`, `test_check_gates_tool_flags_hub_id_usage`, `test_poe_tasks_contain_no_gated_commands`

### perf
`test_latency_budget_text_path_p50_under_300ms`, `test_latency_budget_cas_stage_under_150ms`, `test_book_resolve_p50_under_20ms`, `test_latency_eval_includes_sandbox_respawn_cost`

### e2e / web
`test_pipeline_text_to_result_e2e` ("x karenin türevini al" → `2 x`), `test_pipeline_ink_to_latex_to_result_with_stub_ocr`, `test_pipeline_audio_to_command_with_stub_asr`; `InkCanvas.test.tsx` ("pointer events produce InkML with 2 traces"), `MathView.test.tsx` ("renders \frac; falls back to MathJax for unknown macro; rejects \href"), `wavEncoder.test.ts` ("encodes 16 kHz mono PCM16 header"), `ws.test.ts` ("reconnects and honours slow_down"), `e2e/browsers.spec.ts` ("workspace renders KaTeX result in chromium/firefox/webkit"), `e2e/contribute.spec.ts` ("consent gate then recording accepted"), `e2e/book.spec.ts` ("türev tanımını getir opens BookPanel").

## 7. CPU'da verisiz neler koşar

`gpu`, `data`, `mysql`, `browser` dışındaki her şey; bu, Trainer'ın küçük modelle 2 adım koşması, kontrol noktası/manifest/izleyici, Whisper sarmalayıcısının rastgele küçük yapılandırmayla kurulması, tam 1000 problemlik CAS kıyaslaması (`slow`; gece ve her rapor öncesi) dahildir. Gerçek eğitim (kapı 4) hiçbir testte yoktur.

## 8. Linux/konteyner koşusu — `poe test-linux`

Bu depo Windows'ta yazıldı ve Windows'ta ölçüldü; okul pilotu ise büyük olasılıkla Docker'da
koşacak. Süitin ilk kez bir Linux konteynerinde (2026-09-08, `codex-research-cu130`)
çalıştırılması, hiçbir Windows koşusunun bulamayacağı iki gerçek kusuru ortaya çıkardı:

1. **CAS sandbox'ı Linux'ta tamamen ölüydü.** İşçi başına mutlak 1 GB'lık `RLIMIT_AS`, işçinin
   *tabanının altındaydı*: NumPy'nin OpenBLAS'ı import anında CPU başına 128 MB adres alanı
   ayırır (8 çekirdekte 1032 MB). Isınma `MemoryError` alıyor, `PoolExecutor.start()` 90 s
   zaman aşımına düşüyor ve her CAS isteği `INTERNAL` yanıtlıyordu. (ADR-0016 Ek.)
2. **`Path("D:/aicls-models")` Linux'ta göreli bir yoldur** — tek bileşenli, "D:" adlı bir
   dizin. `tests/unit/ml/test_config.py` ve `test_configs_validate.py` bunu mutlak varsayıyordu;
   `is_absolute()` iddiası orada doğru nedenle değil, `Settings`'in yolu çalışma dizinine göre
   çözmesi sayesinde geçiyordu.

Bunun sessizce tekrar bozulmaması için koşu tek komuta indirildi:

```powershell
poe test-linux                              # Windows'tan: wsl -> docker -> pytest
poe test-linux -- tests/unit/mathengine     # pytest argümanları aynen aktarılır
poe test-linux -- -k sandbox -x
python scripts/run_linux_suite.py --print   # komutu çalıştırmadan yazdır
```

Kapsam ve sınırlar:

- Konteyner argv'si, imaj sabitlemesi, bağlamalar ve onay-kapısı değişkenlerinin aktarımı
  `scripts/gpu/aicls-gpu.sh` içindedir (ADR-0020); `scripts/run_linux_suite.py` ikinci bir
  koşucu değildir, yalnızca o betiğe nasıl ulaşılacağını bilir (Linux'ta doğrudan, Windows'ta
  `wsl -e bash`).
- Varsayılan seçim `poe test-all` ile aynıdır (`gpu`, `data`, `mysql`, `browser` dışarıda;
  `slow`/`perf`/`e2e` **dahil**) ve `--no-gpu` ile koşar: süitte GPU testi zaten seçilmiyor,
  kart eğitimdeyken de koşabilsin diye. Yavaş yarısı dışarıda bırakılsaydı, Hypothesis
  sürümüne duyarlı iki özellik testi (`tests/property/test_asr_audio_io.py`) fark edilmezdi.
- `--timeout=900`: imajın CPU torch'u belirgin biçimde yavaştır (ölçüldü: iki adımlık sentetik
  OCR koşusu konteynerde ~140 s, Windows'ta ~20 s). Deponun 120 s'lik varsayılanı dürüst
  testleri yavaşlıkları yüzünden düşürürdü.
- **GitHub workflow'u yoktur ve olmayacak:** koşu iş istasyonunun WSL2 + nvidia-docker kurulumuna
  bağlıdır, barındırılan bir runner'da böyle bir ortam yoktur. Bu yüzden Linux güvencesi
  otomatik değil, *yerel ve belgelenmiş* bir adımdır: dalları birleştirmeden önce ve her rapor
  öncesi koşulur.
- Hiçbir onay kapısı açılmaz: indirme yok, eğitim yok; `AICLS_GATE_*` konteynere olduğu gibi
  aktarılır, dolayısıyla kapılı bir test içeride de reddeder.

### 8.1 Konteyner koşusunda bilinen ortam gürültüsü (kod kusuru değil)

Aşağıdakiler konteynerin/WSL2'nin özellikleridir; tek tek koşturulduklarında geçerler.
Bir koşuda görülürlerse önce tekrar koşturun, sonra kusur arayın:

- **Saat geriye adım atabiliyor.** Ölçüldü (2026-09-08): aynı süreçte `api.auth.registered`
  `03:55:18.964Z`, hemen ardından gelen istek `03:55:17.960Z` damgasını taşıyor. JWT doğrulaması
  `iat`/`exp` için pay (leeway) tanımadığından, geriye adım atan bir saatte taze bir jeton
  `401 AUTH_INVALID` alır ve `test_edges.py::test_admin_eval_runs_listing_and_lookup` ile
  `test_rbac.py` matrisinden biri düşer. Aynı testler tek başına 172/172 geçer. Okul pilotu bir
  sanal makinede koşacaksa bu yalnızca test gürültüsü değildir: `aicls.api.security` tarafında
  küçük bir `leeway` değerlendirilmelidir (sahibi: api).
- **`git status --porcelain` /mnt üzerinde yavaştır** (ölçüldü: 10,2 s; `DEFAULT_PROCESS_TIMEOUT_S`
  15 s). Paralel koşuda bu sınır aşılabilir ve `aicls.ml.manifest.capture_git_state` o dalda
  `available=True, dirty=False` döndürdüğü için koşu **`citable=True`** işaretlenir — yani
  provenance kontrolü açık tarafa düşer. `test_profile.py::test_profile_report_fields_on_synthetic`
  bunu yakaladı. Düzeltme tek satırdır (durum bilinmiyorsa `available=False`); sahibi: ml.
- **Konteynerin CPU torch'u belirgin biçimde yavaştır**; zamana bağlı iddiaları olan testler
  (`tests/unit/ocr/test_models_torch.py::test_tiny_forward_backward_step_on_cpu`, `< 5 s`)
  yüklü bir makinede sınırı aşabilir. Bunlar makine hızı iddialarıdır, bütçe değil.
