# Signals_big_lab

**Ссылка на датасет**

https://drive.google.com/drive/folders/1tp9B3HaFy3L5WbJA9CNcwIeYrA8VfdBV?usp=sharing

Для использования нужно скачать все 3 папки(below, above, none) и прописать в notebook новые пути.

**Постановка задачи**
* Пусть заданы стол и стул с 5 ножками и ручками зеленого цвета(для контраста с остальными линиями), способный крутиться и менять свою высоту.
Также существует ковер: его мы считаем равномерно примыкающим к столу; по цвету он контрастирует с паркетным покрытием в глубине отверстия стола.
 
* Начальное положение стула однозначно определяется линией ковра: 2 ножки стула стоят на границе коврового покрытия у отверстия в столе.
Считается, что стул можно задвинуть под стол, если эти ножки можно сдвинуть с ковровой линии вглубь отверстия хотя бы на половину длины ножек. Такое возможно и в случае, если ручки стола находятся ниже линии столешницы, так и в случае, если они выше верхней линии. 
 
* Фотографии должны быть сделаны с помощью камеры разрешением не менее 10-ти мегапикселей без цветовой коррекции и понижения яркости(иначе пропадет возможность отличать ручки стула от стола). Не подойдут также фотографии, снятые в черно-белом режиме. На фотографии должны быть видны "опорные" две ножки, которые определяют граничное положение и ручки, по которым определяется высота подъема.

**План решения задачи**
1. Необходимо получить линии столешницы стола – нижнюю и верхние грани. Для этого 
можно использовать детектор Хафа от границ изображения (границы методом canny). 
После применения имеем параметры (h, θ) – перпендикуляр из начала координат и угол 
наклона перпендикуляра, поэтому можно рассчитать высоту прямой (координату y).
Отделить эти линии от прочих можно тем, что они горизонтальные и будут близки к 
середине изображения.
Таким образом, мы получили 2 опорные линии, с помощью которых определяется 
возможность задвинуть стул.

2.  Для определения ручек можно воспользоваться поиском особых точек SIFT. 
Отбираем нужные особые точки по цвету: интересуют зеленые.
Среди полученных точек отбираем те, что имеют наибольшую и наименьшую высоту от 
края изображения.

3. Имеем 2 высоты столешницы: нижний край и верхний край, а также точки ручек, 
соответствующие самому низкому и высокому расположениям.
Можно перейти к сравнению: интересуют случай, когда верхняя точка по высоте меньше
нижнего края столешницы («пройдет ниже») и случай, когда нижняя точка выше верхней 
грани столешницы («пройдет выше»), иначе получили «не пройдет».
Стоит иметь в виду, что на самом деле высота считается от верхнего края.

**Структура репозитория**

Notebook с решением находится в ветке chair-table-feature, там же располагаются директории с репрезентативными примерами.
Прицеплены к реквесту некоторые задачи, которые предполагаются к решению.

**Решение задачи**

1. Для детекции границ столешницы был использован детектор Хафа, однако был изменен способ выделения этих линий - они должны быть почти горизонтальны, но, кроме того, наиболее близки  друг к другу, что позволяет отсечь другие грани стола и иные горизонтальные линии.
2. Для определения ручек: действительно использовался SIFT, затем выделение зеленых среди этих точек.
   Использовались 3 различных метода для попытки определить цвет(ручки):
   * С помощью маски: по соотношению компонент цвета в RGD - известно, что R < G и B < G.
   * С помощью определения основных цветов изображения: кластеризация методом KMeans по 6 основным цветам - число основных цветов подбиралось эмпирически. Оттенки искались в некоторой окрестности найденных основных цветов. Проводилась для каждого изображения по отдельности.
   * Тоже с помощью основных цветов картинки, однако основные оттенки определялись не для каждого изображения по отдельности, а для первого из всех изображений данной директории. Предполагалось, что это приведет ускорению решения.

**Результаты**

Для распознания границ стола точность была удовлетворительной: всего на одном изображении подход наиболее близких друг к другу почти горизонтальных прямых выделил не те 2 прямые.
Однако в процессе решения всплыла некоторая проблема: несмотря на одинаковое освещение, стол и стул располагаются перед окном, поэтому смена угла съемки влияет на оттенок цвета - порой может иметь больший отлив в синие оттенки. 
Сменить положение стола невозможно, к сожалению, поэтому пришлось довольствоваться тем, что точность оказалась несколько ниже предполагаемой.

<ins>Численные результаты:</ins>

Методы: is_green_clast - кластеризация на каждом изображении, is_green_threshold - маска по соотношению компонент цвета, is_green_from_first - кластеризация на первом изображении.
Численные результаты:
| Method              |   Accuracy above |   Accuracy below |   Impossible accuracy |   Total accuracy |
|:--------------------|-----------------:|-----------------:|----------------------:|-----------------:|
| is_green_clast      |              0.3 |              0.3 |                   1   |            0.533 |
| is_green_threshold  |              0.1 |              0.2 |                   0.8 |            0.367 |
| is_green_from_first |              0.4 |              0.5 |                   0.4 |            0.433 |

Видно, что лучшие результаты показывает определение основных цветов изображения на каждом из них. Это оптимальный вариант для решения, поскольку может учитывать тонкости оттенков. Попытки сортировать точки по соотношению цветов между ними были не самыми удачными в силу выше описанного, равно как и кластеризация по цветам в начале.

Кроме того, была попытка использовать hsv модель для определения границ цвета и inRange для отсечения остальных цветов по заданным границам, однако это не привело к должному результату в силу сложностей с определением границ - точность была еще меньше, не стала упоминать среди использованных.

Таким образом, ненадежным способом оказалось опираться на цвет при распознании каких-либо объектов.

**Предположения**

Для дальнейшей разработки можно было бы сделать следующее:
* искать ручки как некоторую компоненту связности или, например, с помощью findContours;
* провести более серьезное исследование влияния числа кластеров при KMeans на точность.
