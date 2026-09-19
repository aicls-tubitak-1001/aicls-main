# Bölüm 1: Limit ve Süreklilik

Bu bölüm tek değişkenli fonksiyonlarda limit kavramını, limit kurallarını ve
süreklilik tanımını tanıtır. Bölümdeki her başlık `aicls.book` etiketleyicisi
tarafından tanınan bir kalıptır; kapsama oranı 1.0 olmalıdır.

## Tanım 1.1: Fonksiyonun Limiti

$f$ fonksiyonu $a$ noktasının delinmiş bir komşuluğunda tanımlı olsun. Her
$\varepsilon > 0$ sayısı için, $0 < |x - a| < \delta$ olduğunda
$|f(x) - L| < \varepsilon$ olacak biçimde bir $\delta > 0$ sayısı bulunabiliyorsa,
$f$ fonksiyonunun $a$ noktasındaki limiti $L$ sayısıdır.

$$\lim_{x \to a} f(x) = L$$

## Teorem 1.2 (Sıkıştırma Teoremi):

$a$ noktasının delinmiş bir komşuluğunda $g(x) \le f(x) \le h(x)$ eşitsizliği
sağlansın. Uç fonksiyonların limitleri aynı sayıya eşitse ortadaki fonksiyonun
limiti de o sayıya eşittir.

$$\lim_{x \to a} g(x) = \lim_{x \to a} h(x) = L$$

## Örnek 1.3: Belirsiz Oran

$0/0$ belirsizliği taşıyan aşağıdaki oran, sıkıştırma yöntemiyle hesaplanır.

$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$

## Tanım 1.4: Süreklilik

$f$ fonksiyonu $a$ noktasında tanımlı, $a$ noktasındaki limiti var ve bu limit
fonksiyonun $a$ noktasındaki değerine eşitse $f$ fonksiyonu $a$ noktasında
süreklidir.

$$\lim_{x \to a} f(x) = f(a)$$

## Teorem 1.5 (Ara Değer Teoremi):

$f$ fonksiyonu $[a, b]$ kapalı aralığında sürekli olsun. $f(a)$ ile $f(b)$
arasındaki her $k$ sayısı için, $f(c) = k$ olacak biçimde en az bir
$c \in (a, b)$ sayısı vardır.

## Şekil 1.6: Süreklilik ve Kopma Noktası

Sürekli bir fonksiyonun grafiği, tanım aralığında kalemi kaldırmadan çizilebilir;
sıçramalı bir fonksiyonun grafiğinde ise kopma noktası görülür.

## Alıştırma 1.7: Limit Hesabı

Aşağıdaki limiti çarpanlara ayırma yöntemiyle hesaplayınız.

$$\lim_{x \to 2} \frac{x^2 - 4}{x - 2}$$
