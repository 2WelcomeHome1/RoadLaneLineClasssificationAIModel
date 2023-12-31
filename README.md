# Алгоритм классификации типа дорожной разметки

Алгоритм классификации типа дорожной разметки — это сверточная нейронная сеть, состоящая из 25 слоев с переходной функцией активации swish. На выходном слое функция активации softmax. Функция потерь categorical crossentropy. В качестве оптимизатора выбран улучшенный алгоритм стохастического градиентного бустинга Adam. На вход поступает цветное изображение размером 252x252 пикселя. 

Для обучения алгоритма классификации дорожной разметки использовался датасет «TuSimple lane classes» 

В качестве показателя эффективности работы алгоритма использовалась метрика wF1-score. Для оценки точности использовались тестовая и валидационная выборки, состоящие из изображений, которые нейросеть ранее не видела. 

Модель классификации типа дорожной разметки на валидационном датасете, состоящем 1332 изображений, показала результат 0.9655, а на тестовом наборе данных, состоящем из 3000 изображений – 0.9587.
