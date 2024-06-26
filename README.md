# Key Tags Extractor

This project is a Python script designed to extract key phrases from textual data using various methods including TF-IDF, TextRank, LDA, Word2Vec embeddings, and BERT embeddings. The script allows you to save the extracted keywords to CSV files or SQLite databases.

### Features

- Extract key phrases using TF-IDF
- Extract key phrases using TextRank
- Extract key phrases using LDA
- Extract key phrases using Word2Vec embeddings
- Extract key phrases using BERT embeddings
- Save results to CSV files
- Save results to SQLite databases

### Requirements

- Python 3.7+
- numpy
- pandas
- scikit-learn
- sumy
- gensim
- transformers
- torch
- nltk

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/key_tags_extractor.git
    cd key_tags_extractor
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

The script can be run from the command line. Below are the commands to extract key phrases using different methods and save the results:

#### Extract and Save to CSV and SQLite

```sh
python main.py <file_path> <n_keywords> --method <method> --csv --sqlite
```

- `<file_path>`: Path to the text file.
- `<n_keywords>`: Number of keywords to extract.
- `<method>`: Method to use for keyword extraction (`tfidf`, `textrank`, `lda`, `word2vec`, `bert`).
- `--csv`: Save results to a CSV file.
- `--sqlite`: Save results to an SQLite database.

#### Example

```sh
python main.py ./data/test_singularity.txt 100 --method tfidf --csv --sqlite
```

This command extracts the top 100 key phrases using TF-IDF from `test_singularity.txt`, saves the results to a CSV file and an SQLite database.

### Results

Below are the top 15 phrases extracted by each method:

#### BERT

| Keyword                               | Score                |
|---------------------------------------|----------------------|
| сингулярности                         | 296.4840364456177    |
| сингулярность                         | 196.83428478240967   |
| например                              | 133.52429962158203   |
| общей теории относительности          | 123.85938148498536   |
| чёрной дыры                           | 117.73581218719484   |
| вращающейся плоскости пыли которая    | 109.88053092956544   |
| теории                                | 106.14102363586426   |
| координат                             | 105.74084281921387   |
| сингулярности называются голыми коническая | 97.14334297180176 |
| дыры                                  | 94.31334686279297    |
| решения уравнений                     | 88.87933826446533    |
| вращающейся чёрной                    | 88.53739471435547    |
| чёрные дыры                           | 87.72338447570802    |
| общей теории                          | 87.33612651824951    |
| чёрных дыр                            | 86.6863416671753     |

#### TF-IDF

| Keyword                               | Score                |
|---------------------------------------|----------------------|
| пространство время                    | 0.08725060159497201  |
| пространства времени                  | 0.0698004812759776   |
| чёрной дыры                           | 0.0698004812759776   |
| вращающейся чёрной                    | 0.052350360956983207 |
| голые сингулярности                   | 0.052350360956983207 |
| общей теории                          | 0.052350360956983207 |
| общей теории относительности          | 0.052350360956983207 |
| решения уравнений                     | 0.052350360956983207 |
| сингулярности могут                   | 0.052350360956983207 |
| теории относительности                | 0.052350360956983207 |
| чёрные дыры                           | 0.052350360956983207 |
| чёрных дыр                            | 0.052350360956983207 |
| большого взрыва                       | 0.0349002406379888   |
| вращающейся чёрной дыре               | 0.0349002406379888   |
| горизонте событий                     | 0.0349002406379888   |

#### LDA

| Keyword                               | Score                |
|---------------------------------------|----------------------|
| сингулярности                         | 0.011504069          |
| сингулярность                         | 0.0106823435         |
| например                              | 0.008217162          |
| координат                             | 0.0057519823         |
| также                                 | 0.0057519823         |
| точке                                 | 0.004930257          |
| является                              | 0.004930257          |
| величины                              | 0.004930257          |
| теории                                | 0.004930257          |
| которых                               | 0.004108532          |
| метрика                               | 0.004108532          |
| это                                   | 0.004108532          |
| точки                                 | 0.004108532          |
| относительности                       | 0.004108532          |
| температура                           | 0.0032868085         |

#### TextRank

| Keyword                               | Score                |
|---------------------------------------|----------------------|
| "Гравитационная сингулярность (иногда сингулярность пространства-времени) — точка (или подмножество) в пространстве-времени, через которую невозможно гладко продолжить входящую в неё геодезическую линию." | 1.0                  |
| "В таких областях становится неприменимым базовое приближение большинства физических теорий, в которых пространство-время рассматривается как гладкое многообразие без края." | 1.0                  |
| "Часто в гравитационной сингулярности величины, описывающие гравитационное поле, становятся бесконечными или неопределёнными." | 1.0                  |
| "К таким величинам относятся, например, скалярная кривизна или плотность энергии в сопутствующей системе отсчёта." | 1.0                  |
| "В рамках классической общей теории относительности сингулярности обязательно возникают при формировании чёрных дыр под горизонтом событий, в таком случае они ненаблюдаемы извне." | 1.0                  |
| "Иногда сингулярности могут быть видны внешнему наблюдателю — так называемые голые сингулярности, например, космологическая сингулярность в теории Большого взрыва." | 1.0                  |
| С математической точки зрения гравитационная сингулярность является множеством особых точек решения уравнений Эйнштейна. | 1.0                  |
| "Однако при этом необходимо строго отличать так называемую «координатную сингулярность» от истинной гравитационной." | 1.0                  |
| "Координатные сингулярности возникают тогда, когда принятые для решения уравнений Эйнштейна координатные условия оказываются неудачными, так что, например, сами принятые координаты становятся многозначными (координатные линии пересекаются) или, наоборот, не покрывают всего многообразия (координатные линии расходятся и между ними оказываются не покрываемые ими «клинья»)." | 1.0                  |
| "Такие сингулярности могут быть устранены принятием других координатных условий, то есть преобразованием координат." | 1.0                  |
| "Примером координатной сингулярности служит сфера Шварцшильда в пространстве-времени Шварцшильда в шварцшильдовских координатах, где компоненты метрического тензора обращаются в бесконечность." | 1.0                  |
| "Истинные гравитационные сингулярности никакими преобразованиями координат устранить нельзя, и примером такой сингулярности служит многообразие в том же решении." | 1.0                  |
| Сингулярности не наблюдаются непосредственно и являются при нынешнем уровне развития физики лишь теоретическим построением. | 1.0                  |
| "Считается, что описание пространства-времени вблизи сингулярности должна давать квантовая гравитация." | 1.0                  |
| Многие физические теории включают математические сингулярности того или иного рода. | 1.0                  |

#### Word2Vec

| Keyword                               | Score                |
|---------------------------------------|----------------------|
| сингулярности                         | 1.0                  |
| сингулярность                         | 1.0                  |
| например                              | 1.0                  |
| координат                             | 1.0                  |
| также                                 | 1.0                  |
| является                              | 1.0                  |
| теории                                | 1.0                  |
| точке                                 | 1.0                  |
| величины                              | 1.0                  |
| относительности                       | 1.0                  |
| это                                   | 1.0                  |
| которых                               | 1.0                  |
| точки                                 | 1.0                  |
| метрика                               | 1.0                  |
| имеют                                 | 1.0                  |

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- This project uses data from [nltk](https://www.nltk.org/), [gensim](https://radimrehurek.com/gensim/), [transformers](https://huggingface.co/transformers/), and [scikit-learn](https://scikit-learn.org/).
- The text data used in this project is from the [Wikipedia Гравитационная сингулярность](https://ru.wikipedia.org/wiki/%D0%93%D1%80%D0%B0%D0%B2%D0%B8%D1%82%D0%B0%D1%86%D0%B8%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B8%D0%BD%D0%B3%D1%83%D0%BB%D1%8F%D1%80%D0%BD%D0%BE%D1%81%D1%82%D1%8C).

