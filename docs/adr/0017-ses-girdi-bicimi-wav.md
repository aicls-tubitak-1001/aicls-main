# ADR-0017: Ses girdisi — tarayıcıda 16 kHz mono PCM16 WAV kodlama; sunucu yalnızca WAV kabul eder; ffmpeg/torchaudio yok

- **Durum:** kabul edildi
- **Tarih:** 2026-09-07
- **Karar vericiler:** Ş. Güler, Ç. Şahin
- **İlgili İP:** İP-2 (veri aktarımı), İP-3 (ses verisi toplama, ASR)

## Bağlam ve problem

`MediaRecorder` tarayıcıya göre webm/opus veya mp4/aac üretir; sunucuda bunları çözmek ffmpeg (harici ikili, iş istasyonunda yok, onaylı kurulum listesinde yok) ya da torchaudio (kapı 3) ister. Kalite denetimi ve eğitim için deterministik, tek biçimli girdi gerekir. Üç hakem de ffmpeg bağımlılığını sorun olarak işaretledi.

## Değerlendirilen seçenekler

1. **webm/opus yükle, sunucuda ffmpeg** — bağımlılık, Windows servis hesabı PATH sorunları, konteyner çeşitliliği.
2. **Tarayıcıda WebAudio ile çözüp `OfflineAudioContext` ile 16 kHz'e yeniden örnekleyip PCM16 WAV kodla (`web/src/lib/wavEncoder.ts`); sunucu `wave` (stdlib) + numpy ile okur, başlığı doğrular, 44,1/48 kHz gelirse numpy windowed-sinc ile 16 kHz'e indirir** — bağımlılıksız, deterministik, QC ve eğitim aynı biçimi görür.

## Karar

Seçenek 2. Kabul: RIFF/WAVE, PCM16, mono/stereo (mono'ya indirgenir), 16/44,1/48 kHz, ≤ 30 s, ≤ 10 MB; başka konteyner 400 `ASR_INPUT_INVALID`/`UPLOAD_INVALID`. Akışlı yol (`/ws/v1/audio`) ham PCM16 250 ms parçalar gönderir. Log-Mel: QC/artırma için numpy; Whisper yolunda HF `WhisperFeatureExtractor` (transformers içinde, numpy tabanlı). `soundfile`/`librosa` de gerekmez. Boyut maliyeti (16 kHz PCM16 = 32 kB/s; 10 s cümle ≈ 320 kB) kabul edilir; toplama sekmesinde yükleme ilerleme çubuğu gösterilir.

## Sonuçlar

- Olumlu: sunucuda sıfır medya bağımlılığı; QC girdisi deterministik; testler PCM16 fikstürlerle.
- Olumsuz: webm'e göre ~5–8× büyük yükleme; eski tarayıcılarda `OfflineAudioContext` yeniden örnekleme farkları → `wavEncoder.test.ts` ve Playwright üç tarayıcıda başlık doğrulaması.

## Öneri metniyle uyum

A.2.3 web arayüzü üzerinden ses toplama, "gürültü giderme ve normalizasyon", log-Mel; İP-2 risk planı "verimli veri formatları kullanılacak" — biçim tek ve deterministik; öneri metni ses konteyneri adlandırmaz.

## Doğrulama

`test_read_wav_rejects_png_bytes_with_wav_extension`, `test_read_wav_accepts_44100_and_resamples_to_16k`, `test_api_asr_transcribe_wav_only`, `wavEncoder.test.ts`, `e2e/contribute.spec.ts`.

## İlgili

[mimari/api.md §3.6, §3.8](../mimari/api.md), [guvenlik-gizlilik.md §6](../guvenlik-gizlilik.md#6-yükleme-ve-girdi-doğrulama), deneme `2026-11-wav-in-browser/`.
