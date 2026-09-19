# Bölüm 2: Türev

Bu bölüm türev kavramını ortalama değişim hızından yola çıkarak tanıtır ve
temel türev alma kurallarını verir. Yapılandırılmış birimler `:::` blokları,
kalan birimler düz "Tanım 2.4:" deseniyle yazılmıştır.

::: tanim 2.1 "Ortalama Değişim Hızı" aliases=["ortalama hız"] page=54
$f$ fonksiyonunun $[a, b]$ aralığındaki ortalama değişim hızı, fonksiyon
değerlerindeki değişimin bağımsız değişkendeki değişime oranıdır.

$$\frac{f(b) - f(a)}{b - a}$$
:::

::: sekil 2.3 "Teğet Doğrusu" page=59
Kesen doğruların limit konumu teğet doğrusudur; teğetin eğimi türevin geometrik
karşılığıdır.
:::

::: tanim 2.4 "Türev" aliases=["türevin tanımı","diferansiyel katsayı"] requires=[tanim-2.1] figure=[sekil-2.3] page=58
$f$ fonksiyonu $a$ noktasının bir komşuluğunda tanımlı olsun. Aşağıdaki limit
varsa bu limite $f$ fonksiyonunun $a$ noktasındaki türevi denir.

$$f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$$
:::

## Teorem 2.5 (Türevlenebilirlik Teoremi):

Bir noktada türevlenebilen fonksiyon o noktada süreklidir. Bunun tersi doğru
değildir; süreklilik türevlenebilirliği gerektirmez.

::: kanit 2.6 "Türevlenebilirlik Teoremi Kanıtı" proves=[teorem-2.5] requires=[tanim-2.4]
Türevin tanımındaki limit varsa, fark çarpanı ayrıştırılarak fonksiyon
değerlerindeki farkın sıfıra gittiği gösterilir.

$$\lim_{h \to 0} \bigl(f(a+h) - f(a)\bigr) = \lim_{h \to 0} h \cdot \frac{f(a+h) - f(a)}{h} = 0$$
:::

::: ornek 2.7 "Kuvvet Kuralı" illustrates=[tanim-2.4] page=62
$f(x) = x^n$ fonksiyonunun türevi, tanım kullanılarak veya kuvvet kuralıyla
hesaplanır.

$$\frac{d}{dx} x^n = n x^{n-1}$$
:::

## Teorem 2.8 (Zincir Kuralı):

$g$ fonksiyonu $x$ noktasında, $f$ fonksiyonu $g(x)$ noktasında türevlenebilir
ise bileşke fonksiyon $x$ noktasında türevlenebilirdir.

$$(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$$

## Sonuç 2.9: Bileşke Fonksiyonun Türevi

Zincir kuralının doğrudan sonucu olarak, dıştaki fonksiyonun türevi ile içteki
fonksiyonun türevi çarpılır.

## Alıştırma 2.10: Zincir Kuralı Uygulaması

Aşağıdaki fonksiyonun türevini zincir kuralıyla hesaplayınız.

$$y = \sin(3x^2 + 1)$$

## Tablo 2.11: Temel Türev Kuralları

| Fonksiyon | Türev |
|---|---|
| $x^n$ | $n x^{n-1}$ |
| $\sin x$ | $\cos x$ |
| $e^x$ | $e^x$ |

## Not 2.12: Sık Yapılan Hata

Bileşke fonksiyonlarda içteki fonksiyonun türevi unutulmamalıdır; bu, öğrencilerin
en sık yaptığı hatadır.
