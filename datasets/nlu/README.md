# `datasets/nlu` — Türkçe komut kümeleri (İP-3 / İP-4)

| Dosya | İçerik | Satır | Kaynak |
|---|---|---|---|
| `reading_scenarios.yaml` | Okuma Senaryoları Kütüphanesi: senaryo etiketi + hedef niyet + zorluk taşıyan Türkçe cümleler | 288 cümle / 19 senaryo | **authored** (proje ekibi) |
| `commands_test.jsonl` | Altın küme: `{id, text, intent, slots, latex_expr, context?, source, tags[]}` | 351 | **authored** (envanter örnekleri + okuma senaryoları) |
| `asr_noisy_tr.jsonl` | Altın kümenin sentetik ASR bozulmalı varyantları | 693 | **authored** (türetilmiş) |
| `generate_datasets.py` | İki JSONL dosyasını yeniden üreten betik (seed 1234) | — | — |

## v1.1 değişiklikleri (altın kümenin açıklaması sıkılaştırıldı)

* **`order` artık her `differentiate` satırında bildirilir.** Bildirilmemiş bir yuvayı yok sayan
  eski `slot_f1` tanımı, üssü türev mertebesi sanan yanlış-cevap hatasını (KNOWN_ISSUES PRIORITY 1)
  ölçüme hiç yansıtmıyordu.
* **Yuva bildiren her satır artık komutun *tamamını* bildirir** (`var`, `direction`, `series.order`,
  `series.point`, `query`). Yeni tanım altında (çözülmüş ama bildirilmemiş yuva = yanlış pozitif)
  eksik açıklamalı bir satır *yanlış* bir satırdır. **Mevcut hiçbir altın değer değiştirilmemiş,
  yalnızca eksikler eklenmiştir**; `test_every_gold_row_with_slots_declares_the_whole_command`
  bunu sürekli denetler.
* **`regression_order` etiketli 12 yeni satır** eklendi: `"x üzeri üçün türevini al"` ailesi ve aynı
  ailenin integral/limit/çözüm/grafik karşılıkları. Bunlar hatanın gerileme takımıdır.

Yeniden üretmek için (depo kökünden, proje venv'i ile):

```
python datasets/nlu/generate_datasets.py
```

## Kaynak/provenans uyarısı (raporlarda aynen belirtilmelidir)

Bu üç dosyanın **tamamı proje ekibince yazılmıştır**; hiçbiri bağımsız bir test kümesi değildir.

* `commands_test.jsonl` iki kaynaktan derlenir: `src/aicls/nlu/inventory/commands.yaml` içindeki
  çalışılmış örnekler (yuva değerleri **elle** yazılmıştır; `slot_f1` yalnızca bunlar üzerinden
  ölçülür) ve `reading_scenarios.yaml` cümleleri (yalnızca altın niyet taşır).
* `asr_noisy_tr.jsonl` gerçek bir ASR çıktısı değildir: `tests/unit/nlu/perturb.py` içindeki
  bozulmalar sentetiktir ve **bozulma karışımının ağırlıkları bir varsayımdır** (henüz ince
  ayarlı bir çözücü yok). Bu kümeden çıkan sayı, ASR'nin gerçek hata dağılımını değil, katmanın
  belirli bozulma tiplerine dayanıklılığını ölçer.
* Dolayısıyla buradan raporlanan `intent_accuracy` / `slot_f1` değerleri **gerileme (regression)
  ölçümüdür**. 2027-05-01 ara raporunda bu ayrım açıkça yazılmalı, sayılar eğitim kapısı
  onaylandıktan sonra gerçek ASR çıktıları (`source: asr_output`) ve pilot verisi
  (`source: pilot`) ile yeniden ölçülmelidir. Bkz. `KNOWN_ISSUES.md`, "Metric provenance".

## Genişletme planı

* `reading_scenarios.yaml` hedefi ≥ 600 cümledir (`docs/ml/veri-yonetimi.md` §1); şu anki küme
  o hedefin çekirdeğidir ve `prompt_sentences` tablosunun tohumudur.
* `kapsam_disi` ve `kapsam_disi_matematik` senaryoları (`intent: none`) sistemin **komut
  üretmemesi** gereken istekleri toplar; `kapsam_disi_matematik` aynı zamanda v1.1 için niyet
  adayı listesidir (tanım kümesi, ters fonksiyon, ekstremum, bileşke, determinant).
* `ifade_okuma` senaryosu (`intent: null`) komut değil, salt matematik telaffuzu örnekleridir;
  altın niyet kümesine girmez, yalnızca ASR istemi olarak kullanılır.
