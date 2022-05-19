# DE-and-SRGP-in-Transfer-Learning

Repository includes implementation of the transfer learning algorithm using differential evolution and genetic programming in symbolic regression
(at the moment, only differential evolution has been added, later genetic programming will be added, taken as open source from another repository).

## Описание

Для запуска программы необходимо установить в среде разработки PyCharm все пакеты, импортируемые исполнительными файлами. Достаточно ограничиться командой во вкладке terminal:

```Python

pip install numpy scipy sympy simplegp
```

## Математическая база метода

  Метод основан на двух мощных алгоритмах стохастической оптимизации - разностной эволюции (DE) и генетическом программировании для символьной регрессии (SRGP). Первый компонент реализован в данном репозитории с обращением к методу DEEP ( Differential Evolution Entirely Parallel method), а в качестве SRGP использовалась реализация алгоритма, написанная Marco Virgolin, расположенная по следующей [ссылке](https://github.com/YaroslavAggressive/SimpleGP)
