# Metin → Komut Katmanı (NLU)

Paket: `aicls.nlu`. Karar: [ADR-0005](../adr/0005-metin-komut-katmani-ip4.md). Öneri metninde bu katman adlandırılmamıştır; A.2.3'teki "Ses verisinden elde edilen metnin uygun komut karşılığı, Bilgisayarlı Cebir Sistemi tarafından icra edilecektir" cümlesinin gerektirdiği bileşendir ve İP-4'ün "OCR ve ses tanıma modüllerinden gelen ... ifadeleri SymPy formatına dönüştürecek ara yazılım modülü" tanımı altında raporlanır. Sahibi Ç. Şahin.

İlke: **kural tabanlı önce, LLM asla sonuç üretmez.** Bu katman yalnızca tipli bir `CasCommand | BookFetch | UiAction` üretir; matematiksel sonuç her zaman SymPy'dan gelir.

## 1. Boru hattı

```
metin (yazılı | ASR çıktısı)
  → normalize_text()           NFC, Türkçe küçük harf (İ→i, I→ı), kesme işareti/ek ayırma, noktalama, dolgu sözcükleri, sayı sözcükleri → rakam, sözlük kanonikleştirme, span izleme
  → IntentMatcher.rank()       envanter desenleri (regex) + anahtar sözcük ağırlıkları → aday niyet listesi
  → SlotExtractor.extract()    yuvalar: EXPR (spoken_to_latex → parse_latex doğrulaması), VAR, BOUNDS, POINT, DIRECTION, ORDER, UNIT_TYPE, QUERY
  → mapper                      niyet + yuvalar → mathengine.commands.* | BookFetch | UiAction
  → resolver                    Matched | Clarification | NoMatch (+ trace)
```

## 2. Komut envanteri biçimi

`src/aicls/nlu/inventory/commands.yaml` (paket kaynağı; `importlib.resources` ile okunur, kurulu wheel'de de bulunur). `inventory/schema.py` içindeki `Inventory` pydantic modeli yükleme anında doğrular; `Inventory.validate_examples()` her örneğin kendi niyetine çözüldüğünü kontrol eder (aynı zamanda regresyon testidir).

```yaml
version: "1.0"
defaults: {var: x, min_score: 0.6, margin: 0.15}
slot_types:
  EXPR:      {kind: latex_expr}          # spoken_to_latex → parse_latex geçmeli
  VAR:       {kind: symbol}
  BOUNDS:    {kind: pair, of: EXPR}      # "sıfırdan bire"
  POINT:     {kind: expr_or_infty}       # "sonsuza", "ikiye", "sıfıra"
  DIRECTION: {kind: enum, values: ["+", "-", "+-"]}   # sağdan / soldan
  ORDER:     {kind: int}                 # "ikinci", "2 mertebeden" (§9.2: çıplak rakam sayılmaz)
  UNIT_TYPE: {kind: enum, values: [Definition, Theorem, Lemma, Corollary, Example, Exercise, Figure, Table]}
  QUERY:     {kind: text}
intents:
  - id: differentiate
    cas_op: diff
    priority: 10
    slots:
      expr:  {type: EXPR, required: true, from_context: last_latex}
      var:   {type: VAR, default: x}
      order: {type: ORDER, default: 1}
    keywords: {türev: 3, türevi: 3, türevini: 3, türevle: 2, diferansiyel: 1}
    patterns:
      - "{expr} (nin|nın|nun|nün|in|ın|un|ün)? türev(i|ini|ni)? (al|hesapla|bul)?"
      - "{expr} (ifadesini|fonksiyonunu)? {var} (ye|ya|e|a)? göre türev(le|ini al)"
      - "{order} (inci|ıncı|uncu|üncü)? (mertebeden|dereceden)? türev(i|ini)? {expr}"
      - "türev(ini)? al {expr}?"
    clarify_tr:
      expr: "Hangi ifadenin türevini alayım?"
      var:  "Hangi değişkene göre türev alayım?"
    examples:
      - {text: "x karenin türevini al", slots: {expr: "x^{2}", var: x, order: 1}}
      - {text: "sinüs x in ikinci türevi", slots: {expr: "\\sin{x}", var: x, order: 2}}
      - {text: "türevini al", context: {last_latex: "x^{3}"}, slots: {expr: "x^{3}"}}
      # ≥ 10 örnek / niyet (şema testi)
  - id: integrate_indefinite   # cas_op integrate; "{expr} in belirsiz integrali", "{expr} integralini al"
  - id: integrate_definite     # cas_op integrate; slots bounds required; "{lower} dan {upper} ye (kadar)? {expr} (nin)? integrali(ni al)?"
  - id: limit                  # slots expr, var, point required, direction; "x {point} (ye|ya|e|a) giderken {expr} (nin)? limiti", "sağdan|soldan"
  - id: solve                  # slots equation; "{expr} denklemini çöz", "{expr} in köklerini bul"
  - id: simplify | expand | factor
  - id: series                 # slots expr, point default 0, order default 6; "{expr} in {point} civarında seri açılımı"
  - id: evaluate_at            # slots expr, subs; "x eşittir iki için {expr} in değeri"
  - id: plot                   # slots expr, range default [-10, 10]; "{expr} in grafiğini çiz", "grafiğini göster"
  - id: book_fetch             # BookFetch; slots unit_type, query; "{query} tanımını getir", "{query} teoremini göster", "örnek 2.7 yi ekrana getir"
  - id: book_search            # BookFetch(unit_type=None); "kitapta {query} ara"
  - id: ui_clear | ui_undo | ui_repeat_last | ui_export    # UiAction; "tahtayı temizle", "geri al", "tekrar", "kaydet"
```

Toplam v1: 17 niyet. `cas_op` değeri `mathengine.commands` modelinin `op` ayırıcısına karşılık gelir; bilinmeyen niyet nesne olarak kurulamaz, dolayısıyla SymPy'a ulaşamaz.

## 3. Sözlük ve sayılar

`inventory/lexicon.yaml` — konuşulan matematik → LaTeX parçası ve ASR varyant katlama:

| Grup | Girdi → çıktı |
|---|---|
| Operatörler | artı `+`, eksi `-`, çarpı `\cdot`, bölü `\frac{}{}`, üssü/üzeri `^{}`, kare `^{2}`, küp `^{3}`, kök/karekök `\sqrt{}`, küp kök `\sqrt[3]{}`, mutlak değer `\lvert \rvert`, faktöriyel `!` |
| Fonksiyonlar | sinüs `\sin`, kosinüs `\cos`, tanjant `\tan`, kotanjant `\cot`, logaritma `\log`, doğal logaritma / ln `\ln`, e üzeri `e^{}`, üstel `\exp` |
| Sabitler | pi `\pi`, sonsuz / artı sonsuz `\infty`, eksi sonsuz `-\infty`, e `e` |
| Yunan | alfa `\alpha`, beta `\beta`, teta `\theta`, lambda `\lambda`, ... |
| Yapı | parantez aç `(`, parantez kapa `)`, paydası / bölü ... , "nin karesi" → `^{2}` |
| ASR varyantları | türef→türev, entegral/integralin/integralini → integral(i), "x kare nin" → "x karenin", "limit al"/"limiti al" |

`inventory/numbers.yaml` + `numbers.py`: sıfır..dokuz, on..doksan, yüz, bin, milyon, buçuk, yarım, çeyrek, virgül; sıra sayıları (birinci, ikinci, 2.); `to_number("yüz yirmi üç") = 123`, `to_number("iki buçuk") = 2.5`. Sözlük katlama **yalnızca NLU içindir**; WER hesabı `aicls.eval.normalizers.tr_v1` ile ayrı ve daha dardır (bkz. [degerlendirme.md](../ml/degerlendirme.md#tr_v1-asr-transkriptleri-için)).

**Sayı öbeği geçerliliği (v1.1).** Türkçe bir sayı öbeği, bir ölçek grubu içinde büyüklük
sınıflarını *azalan* sırada söyler: yüzler → onlar → birler ("iki yüz otuz dört"). `to_number`
artık bu sırayı denetler ve bozuk diziyi toplamak yerine reddeder (`numeral_out_of_order`):
`"dört sıfır"`, `"üç beş"`, `"yirmi yüz"`, `"iki bin milyon"`, `"4 0"` sayı değildir.
Normalizasyon bu reddi *daha kısa* bir öbek deneyerek karşılar (en uzun eşleşme geri izlemesi), bu
yüzden `"x kare eksi dört sıfır"` artık `4` değil `4 0` üretir ve son belirteç kaybolmaz.
Bu, KNOWN_ISSUES PRIORITY 1'in ikinci yarısıydı: yutulan belirteç yüzünden platform söylenmemiş bir
denklemi (`x² − 4`) sessizce çözüyordu.

**Sözcük sınırı kurtarma (v1.1).** `Lexicon.resegment` sözlükte tanınmayan bir belirteci en uzun
eşleşmeyle yeniden böler: `"türevtanımını"` → `türev` + `tanımını`, `"xkarenin"` → `x` + `karenin`,
`"lnx"` → `ln` + `x`. Kurallar dar tutulmuştur, çünkü kapsam dışı bir sözcüğü komut sözcüklerine
bölmek **komut uydurmak** olurdu: (1) yalnızca sözlüğün *anlamadığı* belirteçler denenir; (2) her
parça ya bilinen bir gövde ya da gövde + tek ek olmalıdır; (3) en çok bir parça tek harf (değişken)
olabilir ve bu harf bir buyrum eyleminin (`command_verbs`) hemen yanında duramaz — bu kural olmasa
`"kaç"` → `k` + `aç` diye bölünür ve kapsam dışı bir soru `expand` komutuna dönüşürdü; (4) en çok
`MAX_RESEGMENT_PIECES = 4` parça üretilir.

## 4. Konuşulan matematik dilbilgisi

`spoken_math/grammar.lark` (Lark, LALR), normalize edilmiş token'lar üzerinde:

```
?sum     : product (("artı"|"eksi") product)*
?product : power (("çarpı"|"bölü") power | power)*          # örtük çarpım: "iki x"
?power   : atom ("üssü" atom | "üzeri" atom | "kare" | "küp")?
?atom    : NUMBER | VARIABLE | GREEK | CONST
         | FUNC atom                                         # "sinüs x", "ln x"
         | "kök" atom | "karekök" atom | "küp kök" atom
         | "parantez aç" sum "parantez kapa" | "(" sum ")"
         | "mutlak değer" atom
         | "eksi" atom
```

Çıktı `latex2sympy.ast` düğümleridir; `to_latex` ile LaTeX'e yazılır (`"x kare artı iki x"` → `x^{2} + 2 x`; `"x kare bölü x artı bir"` → `\frac{x^{2}}{x} + 1` — "bölü" yalnızca komşu iki `power`'ı bağlar; tüm paydayı almak için "parantez" ya da "paydası" kalıbı gerekir; kural belgelenir ve örneklerle test edilir). Ayrıştırılamayan aralık ham metin olarak kalır ve yuva `unresolved` işaretlenir; niyet eşleşmesi yine de sürer (açıklama isteğine düşer).

## 5. Niyet ve yuva ayrıştırma

- **Desen derleme:** `{slot}` yer tutucuları tembel yakalama gruplarına, `(a|b)?` grupları isteğe bağlı ek gruplarına derlenir; Türkçe ek varyantları `core.turkish.TR_SUFFIXES` ile üretilir (elle her ek yazılmaz).
- **Puanlama:** `score = w_p·pattern_match (tam 1.0 / kısmi 0.5) + w_k·Σ keyword_weight / max + w_c·token_coverage − w_x·conflicting_keywords`, `priority` eşitlik bozucu. Ağırlıklar `defaults` altında; `datasets/nlu/commands_test.jsonl` üzerinde ızgara aramasıyla bir kez ayarlanır ve ADR notu düşülür.
- **Yuva doğrulama:** `EXPR` → `spoken_to_latex` → `parse_latex` geçmeli; geçmezse doğrudan LaTeX olarak dene (OCR/`last_latex` bağlamı); yine geçmezse yuva geçersiz. `VAR` ifadenin serbest sembollerinden tekilse otomatik, değilse `default`/bağlam/açıklama. `BOUNDS`/`POINT` sayı dilbilgisi + `sonsuz`; `DIRECTION` "sağdan"→`+`, "soldan"→`-`.
- **Karar:** en iyi adayın `score ≥ min_score` **ve** ikinciyle farkı `≥ margin` **ve** tüm zorunlu yuvalar geçerliyse `Matched`; zorunlu yuva eksikse `Clarification(missing_slot, question_tr)`; adaylar yakınsa `Clarification(candidates ≤ 3)` ("Şunu mu demek istediniz: ..."); hiç aday yoksa `NoMatch(suggestions_tr)` — İP-4 risk planı gereği hata yerine yönlendirici mesaj.
- **Hesaplanamayan artık (v1.1, `resolver.unaccounted_content`).** Desen yalnızca *kısmen*
  eşleştiyse cümlenin okunmamış bir parçası vardır. Çoğu zararsızdır (sondaki bir eylem, sözlüğün
  bilmediği bir sözcük) ve orada reddetmek katmanı kırılganlaştırır. İki artık türü zararlıdır ve
  komutu reddettirir: (1) konuşulan matematik dilbilgisinin **kabul ettiği** bir parça — komutun
  düşürdüğü bir ifadedir; (2) **başka bir niyetin en yüksek ağırlıklı anahtar sözcüğü** —
  "x karenin türevini al **ve integralini al**" iki komuttur, birincisini çalıştırıp ikincisini
  atmak sessiz kırpmadır. Sonuç `Clarification` (niyetin zorunlu LaTeX yuvası varsa) ya da
  `NoMatch(reason="unaccounted_tokens")`.
- **Bağlam:** `ParseContext.last_latex` (tahtaya yazılan son ifade) `from_context` yuvalarını doldurur — sınıf akışı: yaz, sonra "türevini al" de. `modality="voice"` ise arayüz her zaman `CommandPreview` onayı ister (ses hatası hesaplamaya sızmaz).
- **İz:** her `Matched`/`Clarification` `trace: list[str]` taşır (normalize edilmiş metin, eşleşen desen, yuva çözümleri); arayüzde "nasıl anladım" paneli ve İP-9 hata analizi için.

## 6. SymPy eşlemesi (`mapper.py`)

| Niyet | Komut | SymPy (mathengine içinde) |
|---|---|---|
| differentiate | `Diff(expr, var, order)` | `diff(expr, var, order)` |
| integrate_indefinite | `Integrate(expr, var)` | `integrate(expr, var)` |
| integrate_definite | `Integrate(expr, var, lower, upper)` | `integrate(expr, (var, lo, hi))` |
| limit | `Limit(expr, var, to, direction)` | `limit(expr, var, to, dir)` |
| solve | `Solve(equation, var)` | `solveset(Eq(lhs, rhs), var, S.Reals)`, boşsa `solve()` |
| simplify/expand/factor | ilgili model | `simplify/expand/factor` |
| series | `Series(expr, var, point, order)` | `series(...).removeO()` |
| evaluate_at | `Evaluate(expr, subs)` | `expr.subs(...).evalf()` |
| plot | `Plot(expr, var, xmin, xmax)` | `mathengine.plotting.render_plot` |
| book_fetch/book_search | `BookFetch(unit_type, query)` | CAS yok; `api.services.pipeline` → `book.resolver` |
| ui_* | `UiAction(action)` | CAS yok; istemciye iletilir |

## 7. Yedek politika (`fallback.py`)

| Durum | Davranış |
|---|---|
| Varsayılan | `IntentFallback` yok. `AICLS_LLM_FALLBACK=false`. `test_llm_fallback_disabled_by_default` bunu doğrular. |
| `AICLS_LLM_FALLBACK=true` | `LlmIntentSuggester.suggest(text, intents) -> str | None` **yalnızca envanterdeki bir niyet kimliği** döndürebilir; ifade dizesi, ses ya da sonuç görmez/üretmez. Dönen niyet kural tabanlı yuva çıkarıcıdan yeniden geçer; yuva doğrulanamazsa `Clarification`. `nlu.source="llm"` işaretlenir, `operations.nlu_source` sütununa yazılır ve değerlendirme raporlarında ayrı sütunda ("with_llm") gösterilir — sözleşme metriklerine karışmaz. |
| Ağ/hata | LLM hatası sessizce yutulur; kural yolu sonucu döner. |

## 8. Değerlendirme ve testler

- Altın küme `datasets/nlu/commands_test.jsonl`: `{id, text, intent, slots, latex_expr, context?, source: "authored"|"pilot"|"asr_output", tags[]}`; ≥ 300 satır (8. hafta), `reading_scenarios.yaml` cümleleri + envanter örnekleri + sonradan gerçek ASR çıktıları. `datasets/nlu/asr_noisy_tr.jsonl`: ek düşürme, sözcük birleştirme, "türef" gibi bozulmalarla üretilmiş varyantlar (`tests/unit/nlu/perturb.py`), hedef niyet doğruluğu ≥ 0.90.
- `aicls eval nlu` → `reports/metrics/nlu_commands_<gitsha>.json`: `intent_accuracy` (hedef ≥ 0.95; İP-2 "komut iletimi >%95" ölçütünün teknik karşılığı), `slot_f1`, `command_exact_match`, `clarification_rate`, kaynak bazında kırılım, hatalı kimlikler. **`slot_f1` tanımı v1.1'de sıkılaştırıldı**: çözülmüş ama altın satırda bildirilmemiş bir yuva artık yanlış pozitiftir (§9.4). Rapora yazılan her sayı hangi tanımla ölçüldüğünü belirtmelidir.
- Testler: `test_nlu_x_karenin_turevini_al_maps_to_diff`, `test_nlu_sifirdan_bire_x_kare_integrali_maps_to_integrate_bounds`, `test_nlu_x_sonsuza_giderken_limit_maps_to_limit_infty`, `test_nlu_sagdan_yaklasirken_sets_direction_plus`, `test_nlu_turev_tanimini_getir_maps_to_book_fetch_definition`, `test_nlu_turevini_al_uses_last_latex_context`, `test_nlu_missing_variable_returns_clarification_tr`, `test_nlu_ambiguous_returns_clarification_with_candidates`, `test_nlu_unknown_phrase_returns_nomatch_with_suggestions`, `test_nlu_normalizer_handles_turkish_i_casefold`, `test_nlu_number_words_yuz_yirmi_uc_to_123`, `test_spoken_math_x_kare_arti_iki_x_to_latex`, `test_spoken_math_bir_bolu_x_to_frac`, `test_property_normalizer_idempotent`, `test_property_parse_command_never_raises_non_aicls_error`, `test_inventory_every_example_resolves_to_own_intent`, `test_inventory_each_intent_has_at_least_10_examples`, `test_nlu_command_test_set_intent_accuracy_above_0_95`, `test_nlu_asr_noisy_set_intent_accuracy_above_0_90`, `test_llm_fallback_disabled_by_default`, `test_llm_fallback_can_only_return_inventory_intent_id`.
- Geliştirici döngüsü: `aicls nlu try "x karenin türevini al"` izi yazdırır; arayüzde `CommandPreview`.

## 9. Uygulama notları

Bu bölüm gerçekleştirilen sürümün tasarım metninden ayrıldığı yerleri ve ölçülen değerleri kaydeder
(sürümler: envanter `1.0`, sözlük `1.0`, sayı tablosu `1.0`).

### 9.1 Sözlük ve normalizasyon kararları

- **Fonksiyon adları LaTeX adlarına indirgenir.** `sinüs`/`sinus`/`sinüsü` → `sin`, `kosinüs` → `cos`,
  `logaritma` → `log`, `doğal log` → `ln`. Böylece hem `spoken_math/grammar.lark` hem de envanter
  desenleri tek bir kanonik biçim görür; §3'teki tablonun ima ettiği "her varyantı ayrı yaz" yükü
  ortadan kalkar.
- **Kanonik LaTeX biçimi yazıcıdan gelir.** §2'deki taslakta `\sin{x}` yazılmıştı; gerçek çıktı
  `aicls.latex2sympy.to_latex`in kanonik biçimidir: `\sin\left(x\right)`. Envanter örnekleri ve
  altın küme bu biçimi kullanır, çünkü ancak o biçim `parse_latex` tarafından yapı gereği kabul
  edilir.
- **Ek ayırma sözlük güdümlüdür.** Bir belirteç yalnızca kalan gövde `lexicon.yaml`'daki
  `vocabulary` listesinde bulunuyorsa bölünür; düzenli ünsüz yumuşaması (`c→ç`, `ğ→k`, `b→p`,
  `d→t`) ve `strip_diacritics` ile aksansız yazım kurtarma (`sifir` → `sıfır`, `esittir` →
  `eşittir`) da denenir. Bu, "kadar" gibi sözcüklerin yanlışlıkla bölünmesini engeller.
- **Ayrık yazılmış ekler ek sayılır.** Hem `x'in` hem de `x kare nin` biçimi tanınır; ek
  belirteçleri `slice_math` tarafından atıldığı için konuşulan matematik dilbilgisine hiç ulaşmaz.
  Bu, İP-3 okuma senaryolarında sık görülen "yavaş dikte" hatasını kapatır ve §3'teki "x kare nin"
  → "x karenin" katlama isteğini karşılar.
- **Ek hoşgörüsü desen derleyicisinde otomatiktir ve tembeldir**: her düz belirteçten sonra en çok
  iki yığılmış ek belirteci (`ifade si nin`) isteğe bağlı olarak yutulur, ancak yalnızca desenin
  geri kalanı başka türlü eşleşmiyorsa. Açgözlü olsaydı `için` sözcüğü `e üzeri x` ifadesindeki
  `e`'yi (yönelme ekiyle aynı yazılır) yutardı.

### 9.2 Envanter ve çözümleme kararları

- **Sondaki `{yuva}?` biçimleri kaldırıldı.** Türkçe SOV'dur; "türevini al x kare" doğal değildir.
  Bunun yerine bağlamdan beslenen ayrı desenler yazıldı (`türev (al|hesapla|bul|çıkar)`), böylece
  isteğe bağlı yuva artık sondaki ek belirtecini yakalayamıyor.
- **Puanlama benimsenen okumaya göre yeniden hesaplanır.** `IntentMatcher.rank` her niyeti *en iyi
  görünen* deseniyle puanlar; yuva çıkarımı başka bir deseni benimserse `matcher.score_candidate`
  puanı o desenle yeniden hesaplar. Aksi hâlde yuvaları tutmayan bir tam eşleşme adayı ayakta
  tutardı.
- **Belirsizlik önceliğe duyarlıdır.** Bir aday, kazananla `margin` içinde *ve* önceliği ondan
  düşük değilse rakip sayılır. `integrate_definite` (öncelik 16) tasarım gereği
  `integrate_indefinite`i (10) yener; `expand` ile `series` ise **bilerek aynı önceliği (8)**
  taşır, çünkü "açılım" Türkçede gerçekten iki anlamlıdır ve doğru davranış soru sormaktır.
- **`var` yuvası CAS niyetlerinde `required: true`'dur.** Çıkarım normal durumlarda çözer (tek
  serbest sembol ya da varsayılan `x` ifadenin içinde); yalnızca çok sembollü ve `x` içermeyen
  ifadelerde açıklama ister. Sessizce `x` varsaymak yanlış türev üretirdi.
- **`VAR` için `mathengine.VAR_PATTERN`'den dar bir kontrol kullanılır**
  (`aicls.nlu.slots.is_variable_name`): tek latin harfi (isteğe bağlı alt indisli) ya da Yunan
  harfi adı. Aksi hâlde "x kare ye göre" kalıbındaki `kare` geçerli bir değişken adı sayılıyordu.
- **Yakalanan ama doğrulanamayan yuva bağlama düşmez.** Kullanıcı yuvayı *söylediyse* ve değer
  geçersizse yuva çözümsüz kalır; sessizce `last_latex`e dönmek kullanıcının sormadığı soruyu
  cevaplamak olurdu.
- **`ORDER`, `DIRECTION` ve `UNIT_TYPE` varsayılan olarak taranır** (`fill: scan`), desenle
  yakalanmaz; Türkçe bunları cümlenin her yerine koyabilir.
- **`ORDER` kuralı v1.1'de daraltıldı (KNOWN_ISSUES PRIORITY 1 — yanlış cevap hatası).** Eski kural
  bir rakamı ardından `mertebe`/`derece`/**`türev`** geliyorsa mertebe sayıyordu. Bu yüzden
  `"x üzeri üçün türevini al"` cümlesindeki `3` **hem** `expr`'e (`x^{3}`) **hem** `order`'a (3)
  yazılıyor, platform `d³/dx³ x³ = 6` hesaplayıp güvenle gösteriyordu; doğrusu `3x²`. Yeni kural üç
  parçalıdır:
    1. `türev` artık bir rakamı çapalamaz (`ORDER_ANCHORS = {mertebe, derece}`); mertebe **söylenmiş**
       olmalıdır — ya bir sıra sayısı sözcüğü (`ikinci`) ya da açık bir `mertebe`/`derece` öbeği.
    2. `üzeri`/`üssü` (`POWER_WORDS`) hemen ardından gelen rakam **üstür**, hiçbir koşulda mertebe
       değildir.
    3. Benimsenen okumada **başka bir yuvanın yakaladığı** belirteçler `ORDER` taramasından düşülür
       (`slots.consumed_indices`); açık `mertebe`/`derece` çapası bunun istisnasıdır, çünkü çapa
       tartışmasız kanıttır ("2 mertebeden türev").
  Envanterde `<NUM>` makrosu ("çıplak rakam") eklendi ve `differentiate`/`series` desenlerinin
  başına `"{expr} <OBJ>? <NUM> (mertebe|derece) …"` kondu; böylece rakamla söylenen mertebe tembel
  `{expr}` yakalamasının içine düşmüyor. Ayrıca mertebe-önce sıralaması için
  `"<ORD> (mertebe|derece)? {expr} <OBJ>? türev …"` deseni eklendi
  ("ikinci mertebeden x karenin türevini al").
- **Yan yana iki çıplak sayı reddedilir** (`spoken_math/transformer.imul`). "iki üç" diye çarpım
  söylenmez; art arda iki sayı ya sayı katlayıcısının okuyamadığı bir dizidir ya da kullanıcının
  sorması gereken bir şeydir. Bu olmadan `"x kare eksi dört sıfır"` sessizce `x² − 4·0`'a dönüşüyordu.

### 9.3 Türkçe metinlerin yeri (kural sapması)

Ortak kural "tüm Türkçe kullanıcı metinleri yalnızca `aicls.core.messages_tr` içinde" der. Bu
katmanda niyete özgü metinler **veri kaynağındadır**: `commands.yaml` içindeki `clarify_tr`,
`label_tr`, `defaults.ambiguous_question_tr`, `defaults.suggestions_tr` ve `lexicon.yaml`. Gerekçe:
(1) §2 bu alanları envanterin parçası olarak tanımlar, (2) envantere yeni bir niyet eklemek kod
değişikliği gerektirmemelidir, (3) bu modül `aicls.core`u düzenleyemez. Genel hata metinleri
(`MSG_NLU_NO_INTENT`, `MSG_PARSE_ERROR_AT`, `MSG_VALIDATION_ERROR`) yine `messages_tr`den gelir.
Durum `aicls.asr.qc.QC_REASONS_TR` ile aynıdır; ileride hepsi birlikte değerlendirilmelidir
(bkz. KNOWN_ISSUES).

`trace` satırları Türkçe değildir; `normalized: …`, `intent: … score=…`, `pattern: …`,
`slot expr='…' (pattern)`, `command: diff` biçiminde makine okunur anahtarlardır. Arayüz "nasıl
anladım" panelinde bunları kendi Türkçe etiketleriyle gösterir.

### 9.4 Ölçülen değerler (v1.1, 2026-09-07, seed 1234)

**Önce metrik tanımı değişti, sonra sayı düştü — sıralama önemlidir.** Eski `slot_f1` tanımı
"altın satırın *bildirdiği*" yuvaları puanlıyor, çözülmüş ama bildirilmemiş yuvayı yok sayıyordu.
`differentiate` satırlarının çoğu `order` bildirmediği için, üssü mertebe sanan hata
(KNOWN_ISSUES PRIORITY 1) `slot_f1 = 1,0000` ve `intent_accuracy = 0,9971` altında **tamamen
görünmezdi**. Yeni tanım: **çözülmüş ama bildirilmemiş yuva yanlış pozitiftir**
(`tests/unit/nlu/metrics.py`, `STRICT_UNDECLARED_IS_FALSE_POSITIVE`).

| Adım | Küme | Satır | `intent_accuracy` | `slot_f1` (P / R) |
|---|---|---|---|---|
| (a) eski tanım, eski küme | yuva bildiren satırlar | 148 | 1,0000 | **1,0000** (1,000 / 1,000) |
| (b) **yeni** tanım, eski küme | yuva bildiren satırlar | 148 | 1,0000 | **0,8000** (0,667 / 1,000) |
| (c) yeni tanım, `order` eklenmiş küme | yuva bildiren satırlar | 160 | 1,0000 | **0,8284** (0,707 / 1,000) |
| (d) yeni tanım, tamamlanmış altın açıklama | yuva bildiren satırlar | 160 | 1,0000 | **1,0000** (1,000 / 1,000) |

(b) ve (c)'deki düşüş **beklenen ve dürüst sonuçtur**: yeni tanım altında eksik açıklamalı bir
altın satır artık *yanlış* bir altın satırdır. (b)→(c) arasındaki 115 yanlış pozitifin tamamı
doğru çözülmüş ama bildirilmemiş varsayılanlardı (`var` 86, `direction` 9, `series.order` 10,
`series.point` 5, `book_fetch.query` 1). (d) bu açıklamayı tamamlar — **hiçbir mevcut altın değer
değiştirilmemiştir, yalnızca eksik olanlar eklenmiştir** (doğrulaması:
`test_every_gold_row_with_slots_declares_the_whole_command`). Böylece `slot_f1` artık "birinin
yazmaya üşenmediği yuvalar doğru" değil, "**komutun her yuvası doğru**" demektir.

**Yeni metrik hatayı gerçekten yakalıyor mu?** Evet: düzeltme öncesi `ORDER` kuralı yeni küme ve
yeni tanım altında `slot_f1 = 0,9894` (P/R 0,989) alır ve dört satırı adıyla gösterir
(`x üzeri üçün türevini al` order 3≠1, `x üzeri beşin…` 5≠1, `x üzeri üç…` 3≠1,
`x üzeri dördün…` 4≠1). Eski tanım + eski küme altında aynı hatalı kural 1,0000 alıyordu.

**v1.1 sürümünün ölçümleri:**

| Küme | Satır | `intent_accuracy` | `slot_f1` | Açıklama isteği oranı |
|---|---|---|---|---|
| `commands_test.jsonl` (tümü) | 351 | **0,9972** | — | 0,000 |
| — yuva altını olan alt küme | 160 | 1,0000 | **1,0000** | 0,000 |
| — kapsam dışı satırlar | 25 | 0,9600 | — | 0,000 |
| `asr_noisy_tr.jsonl` | 693 | **0,9913** | — | 0,001 |

**Bozulma tipine göre gürültülü küme doğruluğu.** Sözcük sınırı kurtarma öncesi/sonrası, *aynı*
dosya üzerinde (dalga öncesi `asr_noisy_tr.jsonl`, 667 satır — böylece karşılaştırma adildir):

| Bozulma | n | önce | sonra |
|---|---|---|---|
| `detach_suffix` | 122 | 1,000 | 1,000 |
| `shorten_suffix` | 110 | 1,000 | 1,000 |
| `misspell` | 106 | 1,000 | 1,000 |
| `capitalise` | 110 | 1,000 | 1,000 |
| `drop_apostrophe` | 24 | 1,000 | 1,000 |
| `insert_filler` | 137 | 0,993 | 0,993 |
| **`join_words`** | 58 | **0,276** | **0,948** |
| küme geneli | 667 | 0,9355 | **0,9940** |

Yeniden üretilen (351 satırdan türetilen, 693 satırlık) gürültülü kümede aynı ölçüm:
`join_words` 0,919 · `capitalise` 0,991 · diğerleri 1,000 · küme geneli 0,9913. İki dosya arasındaki
fark bozulma örneklerinin farklı olmasındandır, katmanın değişmesinden değil.

Kalan `join_words` hataları tek bir sınıfta toplanıyor: birleşen sözcüklerden biri **sözlükte
olmayan bir arama terimi** ("rolleteoremini" → `rolle` + `teoremini`, "kuralıara" → `kural` +
`ara`). Sınır kurtarma sözlük güdümlüdür, dolayısıyla serbest metnin içine uzanamaz; bu bilinçli
bir sınırdır (§9.5).

`test_every_perturbation_type_stays_above_the_floor` her bozulma tipi için 0,85 tabanı koyar:
ortalama, çöken tek bir hata kipini gizler — 0,9355'lik küme ortalamasının 0,276'lık
`join_words`'ü gizlemesi tam olarak buydu.

**Provenans (rapora aynen alınmalıdır).** `commands_test.jsonl` ve `asr_noisy_tr.jsonl` proje
ekibince yazılmıştır; bağımsız test kümesi değil, **gerileme takımıdır**. Gürültülü kümedeki
bozulma karışımının ağırlıkları da ölçüm değil **varsayımdır** (henüz ince ayarlı bir çözücü yok).
Sayılar 2027-05-01 ara raporunda bu çekince ile verilmeli ve eğitim kapısı açıldıktan sonra gerçek
ASR çıktılarıyla yeniden ölçülmelidir. Bkz. `datasets/nlu/README.md` ve KNOWN_ISSUES
"Metric provenance".

### 9.5 Düşmanca tarama (v1.1) ve bilinen sınırlar

`tests/unit/nlu/test_adversarial.py` envanterin **her** çalışılmış örneğini yeniden yazar (dolgu
sözcüğü, kısaltılmış/ayrılmış ek, rakamla yazılmış sayı, Türkçe büyük harf, kesme işareti kaybı;
2000'den fazla varyant) ve *farklı bir komut* üreten hiçbir varyant olmamasını şart koşar. Ayrıca
anlamı değiştiren cümleleri (kuyrukta gizli ikinci komut, düşürülen ifade, söylenmemiş mertebe,
SOV dışı sıralama) besleyip **komut üretilmemesini** doğrular. Bulunanlar:

| Bulgu | Durum |
|---|---|
| Üs sayısının mertebe sanılması (her kardinal, her CAS niyeti) | **Düzeltildi** (§9.2) |
| `"dört sıfır"` gibi bozuk sayı dizisinin belirteç yutması | **Düzeltildi** (§3) |
| "… türevini al **ve integralini al**" — ikinci komutun sessizce atılması | **Düzeltildi** (§5, `unaccounted_content`) |
| "x karenin türevini al **iki y küp**" — düşürülen ifade | **Düzeltildi** (§5) |
| "ikinci mertebeden x karenin türevini al" (mertebe-önce) çözülemiyordu | **Düzeltildi** (yeni desen) |
| `join_words` altında `"kaç"` → `k` + `aç` yanlış bölmesi | **Düzeltildi** (`command_verbs` kuralı) |
| Olumsuzluk: `"türevini alma"` yine `diff` üretir | **Açık** (aşağıya bkz.) |
| 12'den büyük sıra sayıları (`onüçüncü`) tabloda yok | **Açık** (`MAX_ORDER = 12` zaten sınırdır) |
| Sözlükte olmayan arama terimlerinde sınır kurtarma çalışmaz | **Açık, bilinçli** |

- **Olumsuzluk modeli yoktur.** "x karenin türevini alma" / "almayı bırak" hâlâ `diff` üretir.
  Riski düşüktür (kimse bir CAS'a "türev **alma**" demez) ve naif bir `-ma/-me` kuralı sıradan
  adları ("hesaplama", "açılım") bozardı; v1.1 için ayrı bir çalışma gerekir. Test
  `TestKnownLimitations::test_negation_is_not_understood` bu davranışı sabitler ki sessizce
  kötüleşmesin.
- **Fiil-önce (VSO) sıralama bilinçli olarak reddedilir.** "türevini al x karenin" `Clarification`
  döner; Türkçe SOV'dur ve sondaki `{yuva}?` desenleri §9.2'de bilerek kaldırılmıştı.
- **Rakamla yazılmış sıra sayısı ("2. türev") desteklenmez**: normalizasyon noktayı düşürdüğü için
  `"2 türev"` ile ayırt edilemez ve bu ayrım tam olarak yanlış cevap üreten yerdi. Söyleniş:
  "ikinci türev" ya da "2 mertebeden türev".

### 9.5.1 Önceki sürümden devreden sınırlar

- **`tanım kümesi` yanlış pozitifi.** "logaritma x fonksiyonunun tanım kümesini söyle" v1'in 17
  niyetinde yoktur; katman bunu `book_fetch(Definition, "log x fonksiyon")` olarak çözer. CAS'a
  ulaşmadığı için yanlış *sonuç* üretmez, ama v1.1'de bir `domain` niyeti eklenmelidir. Satır
  `datasets/nlu/reading_scenarios.yaml` içinde `kapsam_disi_matematik` etiketiyle bilerek tutulur
  ve ölçüme dâhildir; kapsam dışı doğruluğun 1,000 değil 0,960 olmasının tek nedeni budur.
- **`bölü` kuralı** §4'te belirtildiği gibi yalnızca komşu iki `power`'ı bağlar; tüm paydayı almak
  için "parantez aç … parantez kapa" gerekir. Testle sabitlenmiştir.
- **`aicls eval nlu` komutu ve `reports/metrics/nlu_commands_<gitsha>.json` üretimi bu pakette
  değildir** (sahibi `aicls.eval` + `aicls.cli`). Metrik tanımları ve hesabı şimdilik
  `tests/unit/nlu/metrics.py` içindedir; `EvalReport.as_dict()` doğrudan o JSON'un şeklini verir.
- **`aicls nlu try` komutu** da `aicls.cli` sahasındadır; bu paket kancayı
  `aicls.nlu.resolver.explain(text, context=None) -> list[str]` olarak sunar.
- **Okuma Senaryoları Kütüphanesi 288 cümledir**; öneri metnindeki hedef ≥ 600'dür. Bu sürüm
  çekirdektir ve `aicls.asr.stub.STUB_SENTENCES` içindeki 12 cümlenin tamamını kapsar (sözleşme
  testi: `test_library_covers_the_hard_coded_asr_stub_sentences`).
- **`command_exact_match` metriği hesaplanmamıştır**: altın kümede tam komut nesnesi değil, niyet
  ve yuva değerleri bildirilir. Eklenmesi altın satırlara `command` alanı yazmayı gerektirir.

### 9.6 Sözleşmeye eklenen küçük API'ler

`aicls.nlu.context_from_mapping(mapping)` — YAML/JSON gövdesinden `ParseContext` üretir (API ve
altın küme yükleyicisi için). `spoken_to_latex(text, *, normalized=False)` — `normalized=True`
zaten kanonik belirteç akışı verildiğini söyler ve gereksiz ikinci normalizasyondan kaçınır.
`aicls.nlu.slots.free_symbols` / `is_variable_name`, `aicls.nlu.matcher.score_candidate`,
`Inventory.check_example_counts()` ve `Inventory.examples()` dışa açık yardımcılardır. Spesifikasyondaki
adların hiçbiri değişmemiştir.
