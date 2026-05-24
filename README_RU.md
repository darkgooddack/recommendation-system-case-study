## Содержание

- [Введение](#введение)
- [Проблема](#проблема)
- [Решение](#решение)
- [Семантический поиск](#семантический-поиск)
- [Персональные рекомендации](#персональные-рекомендации)
- [Cold Start](#cold-start)
- [Масштабирование](#масштабирование)
- [Метрики](#метрики)

## 0. Введение
Этот репозиторий посвящён разбору рекомендательной системы, которую мы разрабатывали для платформы [Плюс](https://xn--k1ahh5c.space).

<img width="1072" height="748" alt="image" src="https://github.com/user-attachments/assets/f5a090c3-1c98-4fa5-90cb-3090e07c74cb" />

Это социальная сеть, которая уже включает в себя стену с постами, стримы, видео, чаты и многое другое. Сейчас в разработке находится несколько разделов в том числе и для прослушивания музыки. Помимо этого, на платформе регулярно проходят голосования за новые функции и улучшения — часто прямо на стримах с командой разработки и пользователями.

Я занималась проектированием и реализацией семантического поиска, персональных рекомендаций и обработки пользовательских событий.
Репозиторий будет интересен ML, AI и Python-инженерам, а также тем, кто работает с рекомендательными системами, поиском и высоконагруженными сервисами.

Этот репозиторий будет доступен в нескольких версиях:
- русском 
- английском
- испанском

Это сделано для того, чтобы материал был понятен как русскоязычной, так и международной аудитории, которая работает с ML, рекомендательными системами и backend-архитектурой.

## 1. Проблема

Для такой платформы отсутствие рекомендательной системы приводит к следующим проблемам:
- нерелевантные видео
- мало просмотров
- низкое удержание пользователей

Пользователи не получают релевантный контент. Лента не учитывает поведение и интересы пользователя, из-за чего часть видео остаётся без просмотров, а вовлечённость быстро падает.

Это напрямую влияет на удержание и общую активность внутри продукта. Пользователи смотрят меньше контента, потому что он не попадает в их интересы и целевую аудиторию, а авторы, в свою очередь, получают низкие охваты и просмотры.

## 2. Решение
- Похожие видео
- Персональные рекомендации

Для решения описанных проблем в системе предполагается использование нескольких ключевых компонентов.

Система похожих видео позволяет продолжать пользовательский просмотр в рамках одной темы и увеличивает вовлечённость за счёт релевантных рекомендаций прямо в контексте текущего контента.

Персональные рекомендации формируются на основе поведения пользователя и его взаимодействий с платформой, что позволяет адаптировать ленту под индивидуальные интересы и повышать вероятность просмотра релевантного контента.

## 3. Семантический поиск

### Что такое семантический поиск

Перед тем как обсуждать семантический поиск, стоит кратко затронуть полнотекстовый поиск. Это поиск по словам и тексту с учётом языка, лексем и релевантности результата. Чаще всего такие системы основаны на инвертированном индексе, где ключом выступает слово, а значением список документов или страниц, в которых это слово встречается.

К таким системам относится, например, Elasticsearch. Полнотекстовый поиск хорошо подходит для классических поисковых строк: запрос в поисковой строке Ozon. Также для поисковых строк часто используют поиск с автодополнением через разбиение слов на префиксное дерево. Поиск будет занимать O(len(word)).

Основное преимущество полнотекстового поиска — скорость и простота ранжирования результатов. Система ищет совпадения по словам и сортирует документы по релевантности: количеству совпадений, частоте слов, важности токенов и другим факторам.

Но у такого подхода есть и ограничение: поиск не понимает смысл текста. Одну и ту же вещь можно описать разными словами, через синонимы или просто другими формулировками, и классический полнотекстовый поиск уже может не найти нужный результат. Именно здесь появляется семантический поиск.

**Семантический поиск** — это подход, при котором система пытается искать не по совпадению слов, а по смыслу контента. Для этого текст, видео, описание или другие данные преобразуются в embeddings — векторные представления, отражающие смысл объекта в многомерном пространстве.

Благодаря этому система может находить похожий контент, даже если в нём нет одинаковых слов или прямых совпадений по тексту.

**Преимущества семантического поиска**
- поиск по смыслу, а не по словам
- лучше работает с синонимами и разными формулировками
- более релевантные рекомендации



### Вектора и эмбеддинги

Эмбеддинг — это числовое представление объекта (текста, изображения, видео), которое кодирует его смысл в виде многомерного вектора `"грустный фильм про космос" → [0.12, -0.44, 0.88, ...]` В такой форме данные уже можно сравнивать между собой и искать похожие объекты.

<img width="1230" height="677" alt="image" src="https://github.com/user-attachments/assets/cc5160e5-54ad-41d5-a548-309746a4f95d" />

Сравнить векторы и определить уровень их сходства можно разными способами.

**Косинусное сходство**

$$
\cos(\theta) = \frac{A \cdot B}{\|A\|\|B\|}
$$

Сравнивается угол между векторами, а не их длина.
Идеально для текстовых эмбеддингов, анализа документов и рекомендательных систем. Длина вектора часто зависит от объема данных (например, длины текста), а не от его смысла. Косинусное сходство убирает этот шум.

**Евклидово расстояние**

$$
d(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}
$$

Измеряет физическое расстояние между концами векторов в пространстве по прямой линии. При фиксированной и важной длине векторов, например, в алгоритмах классификации или при распознавании лиц. Фиксирует не только направление, но и разницу в абсолютных значениях признаков. Меньше расстояние — выше сходство.

**Скалярное произведение** 

$$
A \cdot B = \sum_{i=1}^{n} A_i B_i
$$

Просто перемножает соответствующие координаты векторов и складывает результаты. Когда векторы уже нормализованы (их длина равна 1). Это стандарт для быстрых векторных баз данных. Это самая быстрая в вычислениях метрика. Если векторы нормализованы, скалярное произведение математически равно косинусному сходству.


### Transformer модели

Современные эмбеддинги создаются на базе архитектуры **Transformer** (например, BERT, RoBERTa). В отличие от старых алгоритмов, которые кодировали каждое слово отдельно, трансформеры используют механизм **Attention**. 
Модель анализирует предложение целиком. Слово "коса" в контексте "девичья коса" и "острая коса" получит абсолютно разные векторы.


```python
class EmbeddingService:
    def __init__(self, encoder: SentenceTransformer):
        self._encoder = encoder

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = await asyncio.to_thread(
            self._encoder.encode,
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return vectors.tolist()
```

```python
@lru_cache
def get_sentence_transformer() -> SentenceTransformer:
    return SentenceTransformer(
        model_name_or_path=settings.MODEL_NAME,
        device=settings.DEVICE,
    )

def get_embedding_service() -> EmbeddingService:
    encoder = get_sentence_transformer()
    return EmbeddingService(encoder=encoder)
```

В основе сервиса EmbeddingService лежит библиотека `sentence-transformers` (PyTorch и Hugging Face) стандарт для генерации семантических эмбеддингов текстов.

### Архитектура и принципы работы `EmbeddingService`

* **Изолированная обработка**: Метод `encoder.encode()` принимает на вход список строк (`Sequence[str]`). Каждый элемент массива обрабатывается независимо, что полностью исключает смешивание смыслов соседних фраз.
* **Токенизация**: Исходный текст нормализуется и разбивается на токены. Токенизатор автоматически добавляет служебные токены, которые определяют начало и конец предложения, а также разделители. Каждый токен получает уникальный целочисленный идентификатор из словаря модели, после чего формируется набор индексов и маска внимания. Маска внимания - массив из 1 и 0, указывающий нейросети, какие токены являются значимым текстом (1), а какие — пустыми техническими заглушками (0), добавленными для выравнивания длины строк в батче.
* **Векторизация**: Трансформер превращает каждую фразу в плотный вектор фиксированной длины. Размерность вектора определяет его емкость: чем больше чисел в векторе, тем точнее передается сложный контекст и тем больше места он занимает в памяти.
* **Ускорение**: Аргумент `device` автоматически переносит вычисления на GPU, если она доступна. Это ускоряет обработку текстов в десятки раз по сравнению с CPU.

### Выбор модели и альтернативы

Все предобученные модели автоматически скачиваются из репозитория **Hugging Face**. Я выбрала **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**, это очень легкая и быстрая модель, разработанная специально для поиска парафраз. Из-за небольшой размерности экономит оперативную память и обеспечивает максимальную скорость поиска.

> Выбранная модель жестко задает длину массива. Для `MiniLM-L12-v2` это 384. Точно такую же размерность необходимо указать при создании коллекции в векторной базе данных, иначе при попытке сохранить эмбеддинги БД вернет ошибку несовместимости.

### Как видео попадают в ML сервис

Рекомендации строятся в отдельном сервисе, это позволяет разделить нагрузку.

<img width="606" height="256" alt="image" src="https://github.com/user-attachments/assets/f73baf29-3c68-4e14-be68-d5c1d83e0c76" />

```python
class VideoDescriptionPayload(BaseModel):
    video_id: UUID
    description: str
    title: str = Field(max_length=255)
    tags: List[str] = Field(default_factory=list)

    def to_raw_text(self) -> str:
        components = [
            f"Название: {self.title.strip()}",
            f"Описание: {self.description.strip()}",
            f"Теги: {', '.join(t.strip() for t in self.tags if t.strip())}",
        ]

        return "\n".join(block for block in components if not block.endswith(": "))
```

### Milvus

Milvus — это высокопроизводительная векторная база данных с открытым исходным кодом, разработанная специально для обработки, хранения и поиска огромных массивов неструктурированных данных с использованием алгоритмов искусственного интеллекта.

Модели AI преобразуют данные в числовые векторы. Milvus индексирует их и позволяет быстро находить наиболее похожие векторы среди миллиардов других. Это позволяет реализовывать RAG-системы, рекомендательные сервисы и системы распознавания лиц.

**Протокол gRPC и асинхронность**
Класс AsyncMilvusClient под капотом оптимизирован для работы через протокол gRPC. По сравнению с обычным HTTP-интерфейсом (порт 9091), gRPC обеспечивает:
Меньшую задержку (Latency): Бинарный протокол HTTP/2 значительно быстрее передает тяжелые массивы данных (эмбеддинги).
Меньший объем трафика: Векторы сериализуются в Protobuf, что гораздо компактнее текстового JSON.
Полноценную поддержку асинхронности: gRPC нативно поддерживает мультиплексирование запросов в одном сетевом соединении.
Подробнее об асинхронном режиме работы можно прочитать в [официальной документации Milvus по asyncio](https://milvus.io/docs/ru/use-async-milvus-client-with-asyncio.md).

**Параметры инициализации AsyncMilvusClient**
При создании экземпляра AsyncMilvusClient можно гибко настраивать параметры подключения и поведение пула gRPC-каналов:
- **uri** (str) — адрес для подключения к серверу Milvus. Для локального Docker-контейнера обычно используется http://localhost:19530.
- **token** (str) — строка авторизации в формате username:password (по дефолту "root:Milvus"). Если безопасность в кластере отключена MILVUS_COMMON_SECURITY_AUTHORIZATIONENABLED=false (по дефолту отключена), этот параметр можно опустить.
- **db_name** (str) — имя конкретной базы данных внутри Milvus, с которой будет работать клиент (по умолчанию используется "default").
- **timeout** (float) — максимальное время ожидания ответа от сервера (таймаут) для сетевых операций по умолчанию.
- **pool_size** (int) — размер пула соединений. Определяет максимальное количество gRPC-каналов, которые клиент может держать открытыми для одновременной обработки конкурентных асинхронных запросов.

**Управление базой: Milvus Attu**
Для удобного администрирования базы данных на продакшене используется Milvus Attu — это официальный графический интерфейс (Web GUI), открывающийся в браузере. Он выполняет ту же роль, что и pgAdmin для PostgreSQL или Compass для MongoDB.

Через Attu можно визуально управлять коллекциями, создавать индексы, контролировать объем загруженных векторов, настраивать права пользователей и выполнять тестовые поисковые запросы прямо из браузера.

Инструкция по развертыванию и работе с интерфейсом доступна в [руководстве по быстрому старту с Attu](https://milvus.io/docs/ru/quickstart_with_attu.md).

```python
class MilvusConnectionManager:
    def __init__(self):
        self.client: AsyncMilvusClient | None = None
        self._collections = ["item_semantic", "item_behavioral"]

    async def connect(self) -> None:
        if self.client is None:
            self.client = AsyncMilvusClient(
                uri=settings.MILVUS_URL,
                token=settings.MILVUS_TOKEN,
                pool_size=settings.MILVUS_POOL_SIZE,
            )
            logger.info("AsyncMilvusClient connection pool initialized.")
        await self._init_collections()

    async def disconnect(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None
            logger.info("AsyncMilvusClient connection pool closed.")

    async def _init_collections(self) -> None:
        for col_name in self._collections:
            await self._create_collection_if_not_exists(col_name)

    async def _create_collection_if_not_exists(self, name: str) -> None:
        if await self.client.has_collection(collection_name=name):
            logger.info(f"Milvus collection '{name}' already exists.")
            return

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="HNSW",
            params={"M": 8, "efConstruction": 64},
        )

        await self.client.create_collection(
            collection_name=name,
            dimension=settings.VECTOR_DIM,
            primary_field_name="video_id",
            id_type="string",
            max_length=64,
            index_params=index_params,
        )
        logger.info(f"Milvus collection '{name}' initialized successfully.")

milvus_manager = MilvusConnectionManager()
```

```python
def get_milvus_client() -> AsyncMilvusClient:
    if milvus_manager.client is None:
        raise RuntimeError("Milvus connection manager is not initialized.")
    return milvus_manager.client
```

### Поиск похожих видео

**DTO ответа эндпоинта для передачи результатов поиска с весами релевантности.**
```python
class ScoredVideoSearchResult(TypedDict):
    video_id: str
    score: float
```

**Интерфейс работы с векторным хранилищем.**
```python
class IVideoVectorRepository(ABC):
    @abstractmethod
    async def get_vector_by_id(self, video_id: str) -> Optional[list[float]]:
        pass

    @abstractmethod
    async def search_similar_by_vector(
        self, 
        vector: list[float], 
        limit: int
    ) -> list[ScoredVideoSearchResult]:
        pass
```

**Сервисный слой для семантического поиска.**
```python
class SemanticService:
    def __init__(self, vector_repo: IVideoVectorRepository):
        self._vector_repo = vector_repo

    async def get_similar_videos(
        self,
        video_id: str,
        limit: int = 10,
        min_score: float = 0.75,
    ) -> List[ScoredVideoSearchResult]:

        vector = await self._vector_repo.get_vector_by_id(video_id)
        if vector is None:
            return []

        raw_results = await self._vector_repo.search_similar_by_vector(
            vector=vector,
            limit=limit + 1,
        )

        filtered_results = [
            item
            for item in raw_results
            if (
                str(item["video_id"]) != str(video_id)
                and item["score"] >= min_score
            )
        ]

        return filtered_results[:limit]
```

**Репозиторий для работы с данными Milvus.**
```python
class MilvusVideoVectorRepository(IVideoVectorRepository):
    def __init__(self, client: AsyncMilvusClient, collection_name: str):
        self._collection_name = collection_name
        self._client = client

    async def get_vector_by_id(self, video_id: str) -> Optional[List[float]]:
        res = await self._client.get(
            collection_name=self._collection_name,
            ids=[video_id],
            output_fields=["vector"]
        )

        if not res or not isinstance(res, list):
            return None
            
        return res[0].get("vector")

    async def search_similar_by_vector(
        self,
        vector: List[float],
        limit: int,
    ) -> List[ScoredVideoSearchResult]:

        res = await self._client.search(
            collection_name=self._collection_name,
            data=[vector],
            limit=limit,
            output_fields=["video_id"],
            search_params={
                "metric_type": "COSINE",
            }
        )

        if not res or not res[0]:
            return []

        search_results: List[ScoredVideoSearchResult] = []

        for hit in res[0]:
            entity = hit.get("entity", {}) if isinstance(hit, dict) else getattr(hit, "entity", {})
            
            v_id = entity.get("video_id") if isinstance(entity, dict) else getattr(entity, "video_id", None)
            score = hit.get("distance") if isinstance(hit, dict) else getattr(hit, "distance", 0.0)

            if v_id is not None:
                search_results.append({
                    "video_id": str(v_id),
                    "score": float(score),
                })

        return search_results

    async def close(self):
        await self._client.close()
```

## 4. Персональные рекомендации
### Сигналы пользователей
### Пайплайн обработки пользовательских событий
### ALS
### User-Item Matrix
### Implicit feedback
### Обучение модели
### Получение рекомендаций

## 5. Cold Start
### Новые пользователи
### Новые видео
### Популярные видео
### Интересы пользователя
### Георекомендации

## 6. Масштабирование
### Redis кэш
### Batch обновления
### Async processing
### Очереди

## 7. Метрики
### CTR
### Watch Time
### Recall@K
### NDCG

