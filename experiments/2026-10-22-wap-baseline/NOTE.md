# WAP (Zhang ve ark., Pattern Recognition 2017) reposunu ayağa kaldırma — fizibilite ve smoke test

- **Tarih:** 2026-10-22   **Kişi:** Ç. Şahin (Claude Code ile)   **Dal:** exp/wap-baseline   **Git SHA:** 7e7cebd
- **Soru:** `JianshuZhang/WAP` kod tabanı AI-CLS el yazısı matematik tanıma (OCR) için başlangıç noktası olabilir mi? Girdi: ADR-0013 (el yazısı girdi biçimi, karar ~2026-10-30); okuma başlığı "el yazısı matematik tanıma" (`docs/literatur/el-yazisi-matematik-tanima.md`).
- **Kurulum:** WAP `47cce26` (son commit 2020-06-17, 16 commit) scratchpad'e klonlandı; proje ağacına kod kopyalanmadı. Donanım: CPU. Repo içindeki CROHME tarball'ları **açılmadı** ve `.npz` ağırlıkları **yüklenmedi** (DATASET_DOWNLOAD / WEIGHTS_DOWNLOAD defter satırları CROHME'yi ve üçüncü taraf ağırlığı kapsamıyor; ADR-0012). Atılabilir venv: Python 3.11.9 + `Theano==1.0.5`, `numpy<1.24`, `scipy<1.11`. Süre: ~40 dk.
- **Ölçüm:** `aicls.eval` metriği yok — bu bir fizibilite denemesi; ölçülenler derlenebilirlik, bağımlılık kurulabilirliği, sözlük kapsamı.

## Sonuç

| Koşul | Metrik | Değer (GA) | n |
|---|---|---|---|
| Orijinal kaynak, Python 3.11 `py_compile` | derlenen dosya | 2/15 (yalnız `optimizers.py`) | 15 |
| `2to3` sonrası (`VGG/` + `data/` kopyası) | derlenen dosya | 10/10 | 10 |
| `2to3` sonrası kalan Py2/eski API kalıntısı | `scipy.misc.imread/imresize` (scipy ≥ 1.3'te yok) | 1 dosya (`gen_pkl.py`) | — |
| CPU'da model grafiği | `theano.gpuarray.dnn.dnn_batch_normalization_*` zorunlu | **CPU yolu yok** (cuDNN-only BN; `nmt.py` ve `translate.py` import anında düşer) | — |
| Theano 1.0.5, Py 3.11, CPU | `import theano` + basit grafik | kurulum **OK**, import **FAIL** (`numpy.distutils.__config__.blas_opt_info` yok, numpy 1.23.5); g++ yok; `pygpu` için Py3.11 dağıtımı yok | — |
| Lisans | LICENSE dosyası | **yok** (GitHub issue #16 "Adding a license" 2020'de yorumsuz kapanmış) | — |
| Sözlük (`data/dictionary.txt`) | sembol sayısı | 111 (+`<eol>`) | — |
| Sözlük kalkülüs kapsamı | var | `\lim \limits \int \sum \frac \sqrt \infty \rightarrow \cdot \times \pm \sin \cos \tan \log \prime \pi \theta \alpha \beta \leq \geq \neq \exists \forall \in` | — |
| Sözlük kalkülüs kapsamı | **yok** | `\partial \ln \to \prod \left \right \mathrm \cdots`(var) `dx`(ayrı token olarak `d` `x`) | — |
| Eğitim altyazıları (CROHME 2014 train) | ifade / ort. token / maks. token | 8 835 / 15.5 / 96 | 8 835 |
| Test altyazıları (CROHME 2014 test) | ifade / ort. token / maks. token | 986 / 16.1 / 204 | 986 |
| Eğitimde `\lim` / `\int` / `\sum` / `\frac` / `\infty` / `\rightarrow` içeren ifade | adet | 287 / 550 / 552 / 2 494 / 286 / 292 | 8 835 |
| Repo içi veri | tarball | `off_image_train.tar.gz` 8 835 BMP, `off_image_test.tar.gz` 986 BMP, `16.tar.gz` 1 147 BMP (CROHME 2016 test) | — |
| Repo içi ağırlık | `VGG/models`, `DenseNet/models` | `attention_maxlen[200]_dimWord256_dim256.npz` (~41 MB) + `bn_params.npz` | — |

Mimari (koddan): VGG tipi CNN izleyici (4 blok × 4 katman, kanal 32/64/64/128, 3×3), GRU çözücü (256), kapsama (coverage) dikkati (128, 5×5 çekirdek), sözcük gömme 256, hedef sözlük 111, Adam 2e-4, yığın 8, maks. uzunluk 200, maks. görüntü 500 k piksel; ağırlık gürültüsüyle ikinci aşama. README gereksinimi: **CUDA 8.0 + cuDNN v7 + Theano 0.10 + libgpuarray**, Python 2.7 (`cPickle`, `print` ifadesi, `open(..., 'wr')`).

## Yorum

WAP yöntem olarak hâlâ atıf değeri taşır (kapsama dikkatinin HMER'e girişi; SCI, Pattern Recognition 71) ama kod tabanı olarak bu makinede ayağa kalkmaz: Theano 2017'de terk edildi, libgpuarray CUDA 8 ister ve Blackwell (sm_120) için derlenemez; BatchNorm doğrudan cuDNN'e bağlandığı için CPU ile bile ileri geçiş yapılamıyor — smoke test "derlenir mi?" düzeyinde kaldı. Lisans dosyası olmadığından kod varsayılan olarak tüm hakları saklı; proje lisansı da beklemede (ADR-0002) iken kaynak kopyalamak mümkün değil. Modalite de uyuşmuyor: WAP çevrimdışı görüntü modeli, ADR-0013 çevrimiçi izi birincil aday sayıyor. Performans tarafında yazarın kendi bildirdiği tek model ~44–50 % ExpRate (CROHME 2014) 2017 düzeyi; güncel küçük modeller 60–65 %, VLM ince ayarları ~80 %. Projede zaten CNN/ResNet kodlayıcı + transformer çözücü, ışın araması ve iki aile (iz/görüntü) var (`src/aicls/ocr/models/`); WAP'tan alınacak tek somut parça çözücüye isteğe bağlı kapsama dikkati eklemektir — ~50 satır, kod tabanı devralmayı gerektirmez. Göremediklerimiz: ağırlıklar yüklenmediği için paper sayısının yeniden üretimi; CROHME görüntülerinin boyut dağılımı (tarball açılmadı).

## Karar / öneri

- [ ] `src/`'a taşı (PR: —)  - [x] ADR güncelle (ADR-0013 `Girdi:` satırı; yeni ADR: üçüncü taraf kod/lisans politikası)  - [x] **Bırak** (neden: lisans yok; Theano/CUDA 8/Py2; cuDNN-only BN; çevrimdışı; 2017 düzeyi doğruluk)
- Yöntemi referans olarak koru: kapsama dikkati `aicls.ocr.models.decoders` için ablasyon bayrağı adayı (`decoder.coverage: true`); CROHME 2014/2016 veri/istatistikleri kıyas tablosu için.
- Okuma programına dönüş: "el yazısı matematik tanıma" → `docs/literatur/el-yazisi-matematik-tanima.md` (alternatif kod tabanları, veri setleri ve yol haritası).
