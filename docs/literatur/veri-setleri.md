# El Yazısı Matematik / El Yazısı Tanıma Çalışmalarında Kullanılan Açık Veri Setleri

Tarih: 2026-10-22. Kapsam: WAP, TAP, BTTR, CoMER, CAN, SAN, ABM, PosFormer, TAMER, ICAL, Uni-MuMER, TexTeller, UniMERNet, TrOCR, MathWriting, HTR-VT ve DAN çalışmalarında geçen veri setleri.

Doğrulama yöntemi: bağlantılar `curl` (HTTP durum kodu) ve WebFetch ile denendi. Not: Zenodo ve researchdata.edu.au, `curl` isteklerine 403 döndürüyor (bot filtresi), ancak Zenodo sayfaları WebFetch ile başarıyla okundu; bu kayıtlar "doğrulandı" sayıldı (dosya indirmesi ayrıca denenmedi). Sayfa içeriğinden okunmayan, bellekten gelen bilgiler (DOĞRULANMADI) ile işaretlidir.

## 1. Özet tablo

| # | Veri seti | Yıl | Modalite | Boyut | Etiket | Lisans | Erişim | Bağlantı durumu |
|---|---|---|---|---|---|---|---|---|
| 1 | CROHME 2011-2013 (ISICAL paketi) | 2011-2013 | çevrimiçi InkML | ICFHR_package.zip 30,4 MB | InkML + MathML | CC (akademik, ticari değil) | doğrudan indirme | DOĞRULANDI (200) |
| 2 | CROHME 2014/2016/2019 (CoMER data.zip, çevrimdışı) | 2014-2019 | çevrimdışı görüntü + symLg | 30,9 MB | LaTeX caption + symLg | belirtilmemiş (kaynak CROHME: CC) | doğrudan indirme (GitHub) | DOĞRULANDI (200) |
| 3 | CROHME 2023 (Zenodo) | 2023 | InkML + PNG + çift modlu | 1,8 GB | InkML/MathML/LaTeX/SymLG | CC BY-NC-SA 3.0 | doğrudan indirme | DOĞRULANDI (WebFetch) |
| 4 | CROHME Contextual | 2025 | metin bağlamı (JSON) | Source.zip 316,5 MB | hizalı LaTeX + bağlam | CC BY-NC-SA 4.0 | doğrudan indirme | DOĞRULANDI (WebFetch) |
| 5 | HME100K | 2022 | çevrimdışı kamera görüntüsü | 99.109 (74.502 + 24.607); 728,8 MB | LaTeX/sembol (.txt) | hizmet sözleşmesi (ayrıntı DOĞRULANMADI) | 100tal sitesi (akış DOĞRULANMADI) | sayfa 200; indirme akışı DOĞRULANMADI |
| 6 | MathWriting | 2024 | çevrimiçi InkML (+ çevrimdışı render) | ~630k (230k insan + ~400k sentetik); 3,10 GB | LaTeX (normalizedLabel) | CC BY-NC-SA 4.0 | doğrudan indirme | DOĞRULANDI (200, 3.096.141.721 bayt) |
| 7 | MathWriting-human (HF yansı) | 2024 | çevrimdışı render görüntü | 229.864 / 15.674 / 7.644; 1,25 GB | LaTeX | belirtilmemiş | doğrudan indirme | DOĞRULANDI (200) |
| 8 | Uni-MuMER-Data | 2025 | karışım (çevrimdışı görüntü) | 1.613.476 satır; 32,2 GB | LaTeX + Tree-CoT + sayım | MIT (HF kartı) | doğrudan (HF) | DOĞRULANDI (200) |
| 9 | Aida Calculus | 2020 | sentetik "el yazısı" görüntü | 100.000 görüntü | LaTeX + bbox + maske | DOĞRULANMADI | Kaggle (koşul DOĞRULANMADI) | sayfa 200; içerik okunamadı |
| 10 | im2latex-100k | 2016 | basılı (render) | ~100k; Zenodo 306,8 MB | LaTeX | CC0 1.0 | doğrudan indirme | DOĞRULANDI |
| 11 | UniMER-1M / UniMER-Test | 2024 | basılı + ekran görüntüsü + el yazısı | 1.061.791 / 23.757; HF ~6,3 GB | LaTeX | Apache-2.0 (HF) | doğrudan (HF/OpenDataLab) | DOĞRULANDI (200) |
| 12 | TexTeller latex-formulas | 2024 | basılı (render) | 552.340 görüntü (+1.006.245 ham formül) | LaTeX | OpenRAIL | doğrudan (HF) | DOĞRULANDI (200) |
| 13 | TexTeller latex-formulas-80M | 2025 | basılı | ~80M örnek (HF depo ~688 GB) | LaTeX | Apache-2.0 | doğrudan (HF) | DOĞRULANDI (200) |
| 14 | M2E (çok satırlı) | 2023 | çevrimdışı kamera görüntüsü | 99.956 (79.979/9.992/9.985) | LaTeX (satır ayracı `\n`) | CC-BY-NC-4.0 | doğrudan (ModelScope) | DOĞRULANDI (API 200) |
| 15 | MNE (iç içe) | 2024 | çevrimdışı | 3.653 test (N1 1.875, N2 304, N3 1.464) | LaTeX | belirtilmemiş | Google Drive | sayfa 200; indirme DOĞRULANMADI |
| 16 | University-HMER-RealClassroom | 2026 | çevrimdışı sınıf fotoğrafı | 1.636 (1.103/259/274); ~104 MB | tokenlaştırılmış LaTeX | `other` (yer tutucu) | doğrudan (HF) | DOĞRULANDI (API 200) |
| 17 | IAM Handwriting DB | 2002 | çevrimdışı metin | 13.353 satır, 1.539 sayfa | metin | ticari olmayan araştırma | kayıt | siteye bu makineden erişilemedi |
| 18 | READ 2016 | 2016 | çevrimdışı tarihî sayfa | 551,8 MB | PAGE XML | CC BY 4.0 | doğrudan indirme | DOĞRULANDI (WebFetch) |
| 19 | RIMES (ICDAR 2011 satır kesitleri) | 2011 (yayın 2024) | çevrimdışı | 12.111 satır; 382,5 MB | UTF-8 metin | CC BY-NC 4.0 | doğrudan indirme | DOĞRULANDI (WebFetch) |
| 20 | LAM | 2022 | çevrimdışı satır | 25.823 satır | metin | belirtilmemiş | Kaggle | DOĞRULANDI (200) |
| 21 | MathBrush (Waterloo) | 2011 | çevrimiçi iz | DOĞRULANMADI | DOĞRULANMADI | DOĞRULANMADI | e-posta talebi | genel site 200; veri bağlantısı yok |
| 22 | HAMEX | 2011 | çevrimiçi InkML + ses | 4.350 ifade, 58 yazar | InkML/MathML | DOĞRULANMADI | artık erişilemiyor | bağlantı bulunamadı |
| 23 | T-H-E (Türkçe-Macarca-İngilizce karakter) | 2020 | çevrimdışı karakter (28x28) | 156.000 karakter | sınıf etiketi (CSV) | belirtilmemiş | doğrudan (GitHub) | DOĞRULANDI (200) |
| 24 | Türkçe çevrimiçi el yazısı (Bilgin Taşdemir) | 2021 | çevrimiçi iz | 2.041 kelime | DOĞRULANMADI | DOĞRULANMADI | herkese açık bağlantı bulunamadı | - |

## 2. Ayrıntı blokları

### 2.1 CROHME 2011, 2012, 2013 (ISICAL resmî paketi)
- Yıl: 2011-2013. Kaynak çalışmalar: tüm HMER makalelerinin temeli (WAP, TAP, BTTR, CoMER, CAN, ABM, TAMER, ICAL vb.).
- Boyut: ICFHR_package.zip, 30.450.356 bayt (30,4 MB). Test sayıları: 2013 için 671, 2012 için 486 (DOĞRULANMADI, bellekten).
- Modalite: çevrimiçi iz (InkML). Etiket: InkML içinde MathML ve sembol etiketleri; LaTeX türetilir.
- Lisans: sayfada "Creative Commons, akademik ve araştırma amaçlı, ticari kullanım yok".
- Erişim: doğrudan indirme, kayıt yok.
- Bağlantı: https://www.isical.ac.in/~crohme/CROHME_data.html ; https://www.isical.ac.in/~crohme/ICFHR_package.zip (200, application/zip). Not: bu siteyi `curl` ile `-k` olmadan açmak sertifika hatası verdi; `-k` ve WebFetch çalıştı.
- Kalkülüs kapsamı: `\int`, `\sum`, `\lim`, `\frac` sembolleri bulunur (genel bilgi, DOĞRULANMADI); d/dx biçiminde türev yapısı yok, `d` ve `x` düz harf olarak yazılır.
- Bilinen sorunlar: TC11 sitesi (tc11.cvc.uab.es) bu makineden ECONNREFUSED veriyor; ISICAL sayfasında 2014 verisi "planlı" olarak listeleniyor, tam 2014/2016/2019 paketi orada yok.

### 2.2 CROHME 2014 / 2016 / 2019 (çevrimdışı görüntü yansısı)
- En yaygın yol: CoMER deposundaki `data.zip` (https://github.com/Green-Wood/CoMER/raw/master/data.zip, 200, 30.894.119 bayt). PosFormer README'si de aynı dosyayı gösteriyor. WAP deposundaki `data/` klasörü (https://github.com/JianshuZhang/WAP/tree/master/data, 200) `off_image_train.tar.gz`, `off_image_test.tar.gz`, `train_caption.txt`, `test_caption.txt`, `dictionary.txt` içeriyor; hangi yılların olduğu sayfadan okunamadı (DOĞRULANMADI).
- data.zip içeriği (indirilip sayıldı): `train` 8.836 görüntü (caption 8.834 satır), `2014` 986, `2016` 1.147, `2019` 1.199 caption (2019 klasöründe 1.200 görüntü). Her klasörde `img/`, `symLg/`, `caption.txt`.
- Modalite: çevrimdışı görüntü (InkML'den render); etiket: LaTeX token dizisi; ayrıca symLg.
- Lisans: data.zip için ayrı lisans belirtilmemiş; kaynağı CROHME (CC, ticari olmayan).
- Erişim: doğrudan indirme. CAN deposu ise BaiduYun parolalı bağlantı veriyor (kod 1234); Çin dışından erişim sorunlu olabilir (DOĞRULANMADI).
- Bilinen sorunlar: train'de 8.836 görüntü / 8.834 caption, 2019'da 1.200 görüntü / 1.199 caption sayı uyuşmazlığı. CoMER README'si, CROHME 2019 değerlendirmesinin resmî araçlarla yapılmasını söylüyor. Orijinal TC11 sayfaları erişilemedi.
- Çevrimiçi ham InkML için en iyi yedek: CROHME 2023 Zenodo paketi (aşağıda); önceki CROHME toplamalarını içerdiğini belirtiyor (hangi yılların tam olarak bulunduğu DOĞRULANMADI).

### 2.3 CROHME 2023
- Kaynak: ICDAR 2023 CROHME yarışması (DOI 10.5281/zenodo.8428035), 2023-10-10, sürüm 1.
- Dosya: CROHME23.zip, 1,8 GB, MD5 9eb899ddb0e87c80a17f489bf6162c33.
- Modaliteler: çevrimiçi (InkML + MathML + LaTeX + SymLG), çevrimdışı (PNG, taramadan veya InkML render), çift modlu (InkML + PNG). Eğitim kümeleri yapay çevrimiçi veri ve OffHME çevrimdışı görüntü içeriyor. Yeni veri: Nantes, Luleå, Tokyo.
- Lisans: CC BY-NC-SA 3.0. Erişim: doğrudan indirme.
- Bağlantılar: https://zenodo.org/records/8428035 ; https://doi.org/10.5281/zenodo.8428035 (WebFetch ile okundu; curl 403). Dosya adresi: https://zenodo.org/records/8428035/files/CROHME23.zip (curl 403; tarayıcıyla denenmeli).
- Kaynak çalışma: Uni-MuMER (crohme2023 alt kümesi).
- Bilinen sorun: sentetik ve OffHME eklemeleri nedeniyle eğitim dağılımı klasik CROHME'den farklı.

### 2.4 CROHME Contextual Dataset
- Zenodo 10.5281/zenodo.17122781 (2025-09-21), Zouaid, Xie, Mouchère (Nantes). CROHME2019st.json (2,1 MB), CROHME2023Test.json (4,4 MB), Source.zip (316,5 MB). CC BY-NC-SA 4.0. Bağlam destekli formül tanıma içindir; temel HMER karşılaştırmaları için gerekmez.
- Bağlantı: https://zenodo.org/records/17122781 (WebFetch ile doğrulandı).

### 2.5 HME100K
- Yıl: 2022 (SAN makalesi, TAL). Kaynak çalışmalar: SAN, CAN, ABM, ICAL, TAMER, Uni-MuMER, UniMERNet.
- Boyut: sayfaya göre 99.109 görüntü (74.502 eğitim, 24.607 test), 245 sembol sınıfı, 728,8 MB, .jpg + .txt. (Bir arama özetinde 249 sınıf geçti; resmî sayfa 245 diyor, 245 kabul edildi.)
- Modalite: çevrimdışı, kamera ile çekilmiş, on binlerce yazar.
- Lisans: sayfada "Hizmet Sözleşmesi" ve "Gizlilik Politikası" geçiyor; ayrıntı okunamadı (DOĞRULANMADI).
- Erişim: https://ai.100tal.com/dataset (200). "Veri setini indir" düğmesi var; kayıt/form gerekip gerekmediği sayfa metninden anlaşılamadı (DOĞRULANMADI; giriş gerektirmesi muhtemel). Çin dışında erişim sorunu bildirimleri (DOĞRULANMADI). UniMER, telif nedeniyle HME100K'yi elle indirmenizi istiyor.
- Kalkülüs kapsamı: belirgin kalkülüs odağı yok (DOĞRULANMADI).

### 2.6 MathWriting (Google, 2024)
- Makale: arXiv 2404.10690 (230 bin insan, ~400 bin sentetik örnek).
- Dosyalar: tam veri https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz (200, 3.096.141.721 bayt ≈ 2,9 GiB); örnek https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz (200, 1,5 MB). Depo: https://github.com/google-research/google-research/tree/master/mathwriting (README.md, archive_readme.md, Colab not defteri).
- Modalite: çevrimiçi InkML (yalnızca iz, segmentasyon yok; koordinatlar uzayda ve zamanda normalize değil). Dizinler: train, valid, test, symbols, synthetic; ek dosyalar synthetic-bboxes.jsonl, symbols.jsonl. Çevrimdışı render makalede anılıyor.
- Etiket: LaTeX; eğitim için `normalizedLabel` kullanılmalı (symbols dizininde yalnızca `label` var). İfadeler Wikipedia kaynaklı; ~25 bin tanesi programatik üretilmiş karmaşık kesir.
- Lisans: veri CC BY-NC-SA 4.0, kod Apache-2. Erişim: doğrudan indirme.
- Kalkülüs kapsamı: Wikipedia kaynaklı geniş kapsam; integral/toplam/kesir/limit beklenir, ancak sayım yapılmadı (DOĞRULANMADI).
- Bilinen sorunlar: yazar kimliği yok; bazı yazarlar hem train hem valid/test'te yer alıyor (bölme ayrıca ifade bazında yapılmış); koordinatlar cihazlara göre değişken, örnekleme hızı eşit değil.
- HF yansısı: https://huggingface.co/datasets/deepcopy/MathWriting-human (200; görüntü + latex + sample_id + split_tag + data_type; train 229.864, val 15.674, test 7.644; 1,25 GB; lisans alanı boş, orijinalin CC BY-NC-SA 4.0 şartları geçerli sayılmalı; üçüncü taraf yansısı, "yalnızca insan" iddiasının içeriği DOĞRULANMADI).

### 2.7 Uni-MuMER-Data
- https://huggingface.co/datasets/phxember/Uni-MuMER-Data (200). 1.613.476 satır, 32,2 GB, Parquet, 26 alt küme: crohme2023, crohme_train, hme100k_train, mathwriting_train, mathwriting_symbols, im2latex_v2_train ve her biri için `_tree` (Tree-CoT), `_can` (sayım), `_error_find`, `_error_fix` varyantları. Kart lisansı MIT; erişim kapısı yok (gated=false).
- Uyarı: kart MIT diyor, ancak içerdiği kaynak veriler (CROHME ve MathWriting CC BY-NC-SA, HME100K hizmet sözleşmesi) daha kısıtlayıcı; ticari kullanım için hukuki inceleme gerekir.

### 2.8 Aida Calculus Math Handwriting
- https://www.kaggle.com/datasets/aidapearson/ocr-data (200). Sayfa gövdesi araçla okunamadı; bilgiler NeurIPS 2020 özetinden (https://neurips.cc/virtual/2020/20458): 100.000 sentetik üretilmiş "el yazısı" kalkülüs görüntüsü, konu limit ifadeleri, LaTeX + karakter başına bbox + piksel maskesi.
- Boyut (GB), lisans ve indirme koşulu DOĞRULANMADI (Kaggle genellikle giriş ister). Kalkülüs kapsamı: limit (özetle doğrulandı); integral/türev DOĞRULANMADI.
- Bilinen sorun: sentetik; gerçek el yazısına genelleme iddiası makaleye dayanıyor.

### 2.9 im2latex-100k
- Orijinal: Harvard im2markup (https://github.com/harvardnlp/im2markup, 200). Eski `lstm.seas.harvard.edu/latex/data/` adresi artık başka bir siteye (lstmvis.vizhub.ai) yönleniyor; kullanılmamalı.
- Zenodo: https://zenodo.org/records/56198 (WebFetch ile okundu): formula_images.tar.gz 292,2 MB, im2latex_formulas.lst, train/validate/test .lst, toplam 306,8 MB, CC0 1.0, yükleyen Anssi Kanervisto, 2016-06-21.
- HF: https://huggingface.co/datasets/yuntian-deng/im2latex-100k (200): 67.915 satır (55.033/6.072/6.810), 320x64 piksel, CC0.
- Modalite: basılı (render). Kalkülüs: arXiv formülleri, geniş.
- Bilinen sorunlar (HF kartı): 12 eğitim girdisi iki kez geçiyor; bazı dosya adları bölmeler arasında paylaşılıyor; 2022 ön işleme tam belirtilmemiş. Uni-MuMER "im2latex_v2" sürümünü kullanıyor.

### 2.10 UniMER-1M / UniMER-Test
- https://huggingface.co/datasets/wanderkid/UniMER_Dataset (UniMER-1M.zip, UniMER-Test.zip; ~6,3 GB), https://opendatalab.com/OpenDataLab/UniMER-Dataset (200), https://github.com/opendatalab/UniMERNet (200). Apache-2.0 (HF kartı).
- UniMER-1M: 1.061.791 çift = Pix2tex 158.303 + arXiv 820.152 + CROHME train 8.834 + HME100K train 74.502. UniMER-Test: 23.757 (SPE 6.762, CPE 5.921, SCE 4.742, HWE 6.332; HWE = CROHME ve HME100K).
- Önemli: HME100K kısmı telif nedeniyle pakette yok, elle indirilmeli. Modalite: basılı + ekran + el yazısı görüntü; etiket LaTeX.

### 2.11 TexTeller eğitim verisi
- https://huggingface.co/datasets/OleehyO/latex-formulas (200): `cleaned_formulas` 552.340 görüntü-LaTeX çifti (2,78 GB indirme), `raw_formulas` 1.006.245 yalnızca formül; lisans OpenRAIL.
- Güncel: https://huggingface.co/datasets/OleehyO/latex-formulas-80M (200; Apache-2.0; depo ~688 GB, kart "yaklaşık 80 milyon örnek" diyor). Çok büyük; yalnızca gerekirse.
- Modalite: basılı render. El yazısı yok; yalnızca LaTeX formül dili kapsamı için faydalı.
- Kaynak kod: https://github.com/OleehyO/TexTeller (200).

### 2.12 M2E ve MNE (PosFormer)
- M2E: https://www.modelscope.cn/datasets/Wente47/M2E/ (200; ModelScope API ile doğrulandı). 99.956 çok satırlı görüntü, 8:1:1 rastgele bölme (79.979/9.992/9.985), cep telefonu ile sınav kâğıdı ve alıştırma kitabından çekilmiş, etiket JSONL (`name`, `tex`; satırlar `\n` ile ayrılmış), CC-BY-NC-4.0, 1.906 indirme. Kaynak çalışma: "Read Ten Lines at One Glance" (ACM MM 2023), PosFormer.
- MNE: PosFormer README'si Google Drive bağlantısı veriyor: https://drive.google.com/file/d/1iiCxwt05v9a7jQIf074F1ltYLNxYe63b/view (sayfa 200; dosya indirmesinin çalıştığı DOĞRULANMADI). Makaleye göre N1 1.875 + N2 304 görüntü CROHME test kümelerinden, N3 1.464 görüntü yeni toplanmış; toplam 3.653 (makale metni bu toplamı iki farklı biçimde veriyor, DOĞRULANMADI); yalnızca test kümesi. Lisans belirtilmemiş.
- PosFormer kodu: https://github.com/SJTU-DeepVisionLab/PosFormer (200, BSD-2, yalnızca akademik).

### 2.13 University-HMER-RealClassroom (capstone)
- https://huggingface.co/datasets/tuan3110/University-HMER-RealClassroom (API 200, gated=false, ~104 MB). Depo: https://github.com/tuanfptu/SU26AI46_GSU08-Capstone-UniversityHMER (200, MIT).
- 1.636 görüntü: train 1.103, validation 259, test 274. Alanlar: image, sample_id, latex (boşlukla tokenlaştırılmış), category, source, token_count, severity, width, height. ImageFolder + metadata.csv düzeni, `image_checksums.sha256` ve `audit_report.json` var.
- Kalkülüs kapsamı (eğitim): integral 407, derivative_partial 245, limit 178, sum_series 255, log_exp 18. Doğrulama: log_exp 125 + mixed_nested 134. Test: yalnızca mixed_nested 274. Yani doğrulama ve test dağılımları eğitimden ve birbirinden farklı.
- Lisans: HF kartında `license: other`; kart, açık bir lisansla değiştirilmeden herkese açık yayın yapılmaması gerektiğini ve ilk sürümün "private" olduğunu söylüyor. Yine de API anonim erişime izin verdi (kart ile erişim durumu çelişiyor). GitHub deposu MIT diyor. Kullanmadan önce sahiplerle netleştirin.
- Bilinen sorunlar: ölçek küçük; doğrulama/test dağılım kayması; integral/toplam sınırları sıklıkla çapraz veya sağda yazılmış, dikey yığın ("display") sınırlar az.
- Kart içinde kayıtlı sonuçlar (tekrar üretilmedi): TAMER özgün test ExpRate %4,38; TAMER A3 RealFT %71,17; Uni-MuMER LoRA %74,82; Uni-MuMER sıfır atış %23,72.

### 2.14 IAM, READ 2016, RIMES, LAM (HTR-VT, DAN, TrOCR için)
- IAM Handwriting DB: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database ve indirme sayfası https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database. Bu makineden ECONNREFUSED; yalnızca arama sonuçları ve HTR-VT README'si doğrulandı. Kayıt zorunlu, ticari olmayan araştırma lisansı; 657 yazar, 1.539 sayfa, 13.353 satır, 115.320 kelime (arama özeti). Çevrimiçi sürümler (IAM-On, IAMonDo) ayrı sayfalarda. TrOCR'ın eğitim verileri (IAM, sentetik vb.) bellekten (DOĞRULANMADI).
- READ 2016: https://zenodo.org/records/1164045 (CC BY 4.0; Train-And-Val 493,2 MB + Test 58,6 MB). 400 eğitim sayfası, PAGE XML satır düzeyi, Erken Modern Almanca. Test kümesini içeren tam sürüm: https://zenodo.org/records/1297399 (içerik yalnızca arama özetinden, DOĞRULANMADI).
- RIMES: ICDAR 2011 satır kesitleri https://zenodo.org/records/10805048 (CC BY-NC 4.0; RIMES-2011-Lines.zip 382,5 MB; 12.111 satır görüntüsü; 10.188/1.138/778; Mitek 2024'te yayımladı). Teklia'nın RIMES sayfası 404 verdi. DAN'ın kullandığı RIMES 2009 sürümünün indirme yeri DOĞRULANMADI.
- LAM: https://www.kaggle.com/datasets/vpippi/lam-dataset (200), proje sayfası https://aimagelab.ing.unimore.it/go/lam (200), 25.823 satır; temel bölme 19.830/2.470/3.523; lisans belirtilmemiş.
- Kod depoları: https://github.com/YutingLi0606/HTR-VT , https://github.com/FactoDeepLearning/DAN (200).
- Kalkülüs: yok; yalnızca genel el yazısı ön eğitimi için.

### 2.15 MathBrush ve HAMEX
- MathBrush (Waterloo 2011 külliyatı): sistem sitesi https://www.scg.uwaterloo.ca/mathbrush/ (200). Veri indirme bağlantısı yok; erişim yalnızca ekibe e-posta ile talep (arama özeti; DOĞRULANMADI). Boyut ve etiket biçimi DOĞRULANMADI.
- HAMEX: 4.350 çevrimiçi el yazısı ifade, 58 yazar, Fransızca konuşma kaydı; InkML + sembol segmentasyonu + MathML. Arama sonuçlarına göre artık herkese açık değil; doğrulanabilir bağlantı bulunamadı.

### 2.16 Türkçe el yazısı
- T-H-E: https://github.com/bartosgaye/thedataset (200). 200 katılımcı, 78 sınıf, 156.000 karakter (28x28 ikili CSV), altı sürüm (II: Türkçe özel karakterler 24.000/12 sınıf; IV: İngilizce 104.000/52 sınıf; VI: artırmasız özgün 78.000). Atıf: Acta Polytechnica Hungarica 2020. Lisans belirtilmemiş. Karakter düzeyi; matematik içermez.
- Bilgin Taşdemir (Sabancı Üniversitesi / İstanbul Medeniyet Üniversitesi): çevrimiçi Türkçe kelime tanıma; sentetik veri + transfer öğrenmesi, 2.041 gerçek Türkçe kelime üzerinde test (karakter doğruluğu %61'den %88'e). Tez: https://research.sabanciuniv.edu/36482/1/10198291_Esma_Fatima_Bilgin_Tasdemir.pdf . Veri setinin herkese açık sunulduğuna dair kanıt yok (DOĞRULANMADI). Ön eğitim için UNIPEN ve IAM-On kullanılmış (arama özeti).

## 3. Kalkülüs notasyonu kapsamı: kısa karşılaştırma

| Veri seti | lim | int | sum | frac | d/dx türev |
|---|---|---|---|---|---|
| CROHME | var (sayım DOĞRULANMADI) | var | var | var | yok (düz harf) |
| MathWriting | beklenir (DOĞRULANMADI) | beklenir | beklenir | var (~25k sentetik karmaşık kesir) | beklenir (DOĞRULANMADI) |
| Aida | var (limit odaklı) | DOĞRULANMADI | DOĞRULANMADI | DOĞRULANMADI | DOĞRULANMADI |
| RealClassroom | 178 | 407 | 255 | karışık | 245 (derivative_partial) |
| HME100K, M2E, UniMER | genel | genel | genel | genel | belirgin değil |

## 4. Doğrulanamayan bağlantılar

- https://tc11.cvc.uab.es/datasets/ ve /CROHME-2014_2 : bağlantı reddedildi (ECONNREFUSED, 158.109.8.45). CROHME 2014/2016/2019 resmî kopyası doğrulanamadı.
- https://fki.tic.heia-fr.ch/ (IAM) : bu makineden erişilemedi.
- https://www.a2ialab.com/doku.php?id=rimes_database:start : erişilemedi; https://www.teklia.com/en/research/rimes-database : 404.
- https://zenodo.org/records/* dosya adresleri (CROHME23.zip, 1297399 vb.): curl 403; sayfa içeriği yalnızca bazılarında WebFetch ile okundu, dosya indirmesi denenmedi.
- https://researchdata.edu.au/crohme-competition-recognition-expressions-png/639782 : 403 (CROHME PNG yansısı; içerik okunamadı).
- https://ai.100tal.com/dataset : sayfa 200 ancak indirme akışı (giriş/form) test edilmedi.
- https://www.kaggle.com/datasets/aidapearson/ocr-data : 200 ancak sayfa gövdesi okunamadı (boyut, lisans DOĞRULANMADI). Denenen kaggle.com/datasets/antoniorosano/lam-dataset 404 verdi (doğru adres vpippi/lam-dataset).
- https://drive.google.com/file/d/1iiCxwt05v9a7jQIf074F1ltYLNxYe63b/view (MNE) : açılış sayfası 200, dosya indirmesi doğrulanmadı.
- http://lstm.seas.harvard.edu/latex/data/ : başka siteye yönleniyor (kullanma).
- https://github.com/Bandi-yu/ABM, https://github.com/qingzhenduyu/CoMER, https://github.com/dmartin-isi/DAN : 404 (tahmin edilen depo adları yanlış; ABM'nin gerçek deposu bulunamadı).
- https://pan.baidu.com/s/1qUVQLZh5aPT6d7-m6il6Rg (CAN, CROHME) : denenmedi.
- https://www.v7labs.com/open-datasets/aida : https://www.v7darwin.com/open-datasets adresine yönleniyor, o adres 404.
- MathBrush ve HAMEX için indirme bağlantısı yok; Taşdemir Türkçe çevrimiçi veri seti için herkese açık bağlantı bulunamadı.

## Kaynaklar

- https://www.isical.ac.in/~crohme/CROHME_data.html
- https://www.isical.ac.in/~crohme/ICFHR_package.zip
- https://github.com/JianshuZhang/WAP/tree/master/data
- https://github.com/Green-Wood/CoMER ; https://github.com/Green-Wood/CoMER/raw/master/data.zip
- https://github.com/vndee/offline-crohme
- https://zenodo.org/records/8428035 ; https://zenodo.org/records/17122781 ; https://zenodo.org/records/56198 ; https://zenodo.org/records/1164045 ; https://zenodo.org/records/10805048
- https://github.com/google-research/google-research/tree/master/mathwriting ; https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz ; https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz ; https://arxiv.org/abs/2404.10690
- https://huggingface.co/datasets/deepcopy/MathWriting-human ; https://huggingface.co/datasets/phxember/Uni-MuMER-Data ; https://huggingface.co/datasets/yuntian-deng/im2latex-100k ; https://huggingface.co/datasets/wanderkid/UniMER_Dataset ; https://huggingface.co/datasets/OleehyO/latex-formulas ; https://huggingface.co/datasets/OleehyO/latex-formulas-80M ; https://huggingface.co/datasets/tuan3110/University-HMER-RealClassroom
- https://ai.100tal.com/dataset ; https://github.com/tal-tech/SAN ; https://github.com/LBH1024/CAN
- https://www.kaggle.com/datasets/aidapearson/ocr-data ; https://neurips.cc/virtual/2020/20458
- https://github.com/SJTU-DeepVisionLab/PosFormer ; https://www.modelscope.cn/datasets/Wente47/M2E/ ; https://arxiv.org/html/2407.07764v1
- https://github.com/tuanfptu/SU26AI46_GSU08-Capstone-UniversityHMER
- https://github.com/bartosgaye/thedataset
- https://github.com/opendatalab/UniMERNet ; https://opendatalab.com/OpenDataLab/UniMER-Dataset ; https://github.com/OleehyO/TexTeller
- https://github.com/YutingLi0606/HTR-VT ; https://aimagelab-legacy.ing.unimore.it/imagelab/page.asp?IdPage=46 ; https://www.kaggle.com/datasets/vpippi/lam-dataset ; https://github.com/aimagelab/LAM ; https://arxiv.org/abs/2208.07682
- https://github.com/FactoDeepLearning/DAN ; https://github.com/harvardnlp/im2markup
- https://www.scg.uwaterloo.ca/mathbrush/
- https://earsiv.medeniyet.edu.tr/items/68892d0d-2c63-42fc-b218-323d79d1e175/full ; https://research.sabanciuniv.edu/36482
