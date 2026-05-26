# recommendation-system-case-study

## Contents

* [Introduction](#introduction)
* [Semantic Search](#semantic-search)
* [Personalized Recommendations](#personalized-recommendations)
* [Cold Start](#cold-start)
* [Metrics](#metrics)

## Introduction
This repository is dedicated to the analysis of the recommendation system developed for the [Plus](https://xn--k1ahh5c.space) platform.

<img width="1072" height="748" alt="image" src="https://github.com/user-attachments/assets/f5a090c3-1c98-4fa5-90cb-3090e07c74cb" />

This is a social network that already includes a feed with posts, streams, videos, chats, and much more. Several sections are currently under development, including a music streaming feature. In addition, the platform regularly hosts votes for new features and improvements — often directly during live streams with the development team and users.

I was responsible for designing and implementing user event processing, semantic search, and personalized recommendations.
The repository will be of interest to ML, AI, and Python engineers, as well as those working with recommendation systems, search, and high-load services.

This repository will be available in several versions:
- Russian 
- English
- Spanish

This is done to ensure the material is accessible to both Russian-speaking and international audiences working with ML, recommendation systems, and backend architecture.

## Problem

For a platform like this, the absence of a recommendation system leads to the following issues:
- Irrelevant videos
- Low view counts
- Low user retention

Users do not receive relevant content. The feed does not account for user behavior and interests, causing some videos to get no views, while engagement drops rapidly.

This directly affects retention and overall activity within the product. Users watch less content because it does not align with their interests and target audience, while creators, in turn, get low reach and view counts.

## Solution
- Similar videos
- Personalized recommendations

To address these issues, the system incorporates several key components.

The similar videos system allows users to continue watching content within the same topic, increasing engagement through relevant recommendations directly in the context of the current content.

Personalized recommendations are generated based on user behavior and interactions with the platform. This adapts the feed to individual interests and increases the likelihood of users watching relevant content.
## Semantic Search

### What is Semantic Search

Before discussing semantic search, it is worth briefly touching upon full-text search. This is a word- and text-based search method that accounts for language, tokens, and result relevance. Most often, such systems are based on an inverted index, where the key is a word and the value is a list of documents or pages where this word occurs.

Systems of this type include, for example, Elasticsearch. Full-text search is well-suited for classic search queries. Additionally, search inputs often utilize autocomplete features powered by prefix trees (tries) to split words. The search operation takes O(len(word)).

The main advantage of full-text search is its speed and simplicity in ranking results. The system searches for exact word matches and sorts documents by relevance based on factors like match count, term frequency, token importance, and other metrics.

However, this approach has a significant limitation: the search does not comprehend the meaning behind the text. The same concept can be described using different words, synonyms, or alternative phrasing, which may prevent classic full-text search from finding the correct result. This is exactly where semantic search comes in.

**Semantic search** is an approach where the system attempts to search based on the meaning of the content rather than word matches. To achieve this, text, video, descriptions, or other data are transformed into embeddings — vector representations that reflect the semantic meaning of an object in a multidimensional space.

This enables the system to discover similar content even when there are no identical words or direct textual matches.

**Advantages of Semantic Search**
- Searches by meaning, not just words
- Handles synonyms and alternative phrasing much better
- Delivers more relevant recommendations
  
### Вектора и эмбеддинги

An embedding is a numerical representation of an object (text, image, video) that encodes its meaning as a multidimensional vector: `"sad movie about space" → [0.12, -0.44, 0.88, ...]`. In this form, data can be compared with one another, and similar objects can be identified.

<img width="949" height="525" alt="image" src="https://github.com/user-attachments/assets/1d0408c3-6e99-476a-ab29-de749bd0ffd5" />

Vectors can be compared, and their level of similarity determined, in various ways.

**Cosine similarity**

$$
\cos(\theta) = \frac{A \cdot B}{\|A\|\|B\|}
$$

It compares the angle between vectors, rather than their length.
It is ideal for text embeddings, document analysis, and recommendation systems. Vector length often depends on the volume of data (e.g., text length) rather than its semantic meaning; cosine similarity eliminates this noise.

**Euclidean distance**

$$
d(A, B) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}
$$

It measures the physical distance between the endpoints of vectors in space along a straight line. It is employed in contexts where the length of the vectors is fixed and significant—for instance, in classification algorithms or facial recognition. It captures not only the direction but also the difference in the absolute values ​​of the features. The shorter the distance, the greater the similarity.

**Scalar product** 

$$
A \cdot B = \sum_{i=1}^{n} A_i B_i
$$

It simply multiplies the corresponding coordinates of the vectors and sums the results. This applies when the vectors are already normalized (i.e., their length equals 1). This is the standard for fast vector databases. It is the computationally fastest metric available. If the vectors are normalized, the dot product is mathematically equivalent to cosine similarity.

### Transformer model

Modern embeddings are built upon the **Transformer** architecture (e.g., BERT, RoBERTa). Unlike older algorithms, which encoded each word separately, Transformers utilize the **Attention** mechanism.
The model analyzes the sentence as a whole. The word "kosa"—in the contexts of "maiden's braid" and "sharp scythe"—will receive entirely different vectors.

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

The EmbeddingService is built upon the `sentence-transformers` library (PyTorch and Hugging Face)—the standard for generating semantic text embeddings.

### Architecture and principles of operation of `EmbeddingService`

* **Isolated Processing**: The `encoder.encode()` method accepts a list of strings (`Sequence[str]`) as input. Each element in the array is processed independently, completely preventing the mixing of meanings between adjacent phrases.
* **Tokenization**: The source text is normalized and split into tokens. The tokenizer automatically appends special tokens that define the start and end of a sentence, as well as delimiters. Each token is assigned a unique integer identifier from the model's vocabulary, which is then used to form a set of indices and an attention mask. The attention mask is an array of 1s and 0s indicating to the neural network which tokens represent meaningful text (1) and which are empty technical padding tokens (0) added to align string lengths within a batch.
* **Vectorization**: The transformer converts each phrase into a dense vector of fixed length. The dimension of the vector determines its capacity: the more numbers in the vector, the more accurately complex context is captured, and the more memory it requires.
* **Acceleration**: The `device` argument automatically offloads computations to the GPU if available. This accelerates text processing tens of times compared to CPU-based execution.

### Model selection

All pre-trained models are automatically downloaded from the **Hugging Face** repository. I selected **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**—a very lightweight and fast model designed specifically for paraphrase retrieval. Thanks to its compact dimensionality, it conserves RAM and ensures maximum search speed.

> The selected model strictly defines the array dimension. For `MiniLM-L12-v2`, this value is 384. You must specify this exact same dimension when creating a collection in your vector database; otherwise, the database will return a compatibility error when you attempt to save the embeddings.

### How Videos Enter the ML Service

Recommendations are generated in a separate service, which allows for load distribution.

<img width="728" height="273" alt="image" src="https://github.com/user-attachments/assets/831e4a3e-abd4-4a04-be31-a44093720064" />

```python
class VideoDescriptionPayload(BaseModel):
    video_id: UUID
    description: str
    title: str = Field(max_length=255)
    tags: List[str] = Field(default_factory=list)

    def to_raw_text(self) -> str:
        components = [
            f"Name: {self.title.strip()}",
            f"Description: {self.description.strip()}",
            f"Tags: {', '.join(t.strip() for t in self.tags if t.strip())}",
        ]

        return "\n".join(block for block in components if not block.endswith(": "))
```

### Milvus

Milvus is a high-performance, open-source vector database designed specifically for processing, storing, and searching massive arrays of unstructured data using artificial intelligence algorithms.

AI models transform data into numerical vectors. Milvus indexes these vectors, enabling the rapid retrieval of the most similar vectors from among billions of others. This capability facilitates the implementation of RAG systems, recommendation engines, and facial recognition systems.

**The gRPC Protocol and Asynchrony**
Under the hood, the `AsyncMilvusClient` class is optimized to operate via the gRPC protocol. Compared to the standard HTTP interface (port 9091), gRPC offers:
- **Lower Latency:**
  The binary HTTP/2 protocol transmits large data arrays significantly faster.
- **Reduced Traffic Volume:**
  Vectors are serialized using Protobuf, which is much more compact than text-based JSON.
- **Full Asynchronous Support:**
  gRPC natively supports request multiplexing over a single network connection.

You can read more about asynchronous mode in the [official Milvus documentation on asyncio](https://milvus.io/docs/use-async-milvus-client-with-asyncio.md).

**AsyncMilvusClient Initialization Parameters**

When creating an instance of AsyncMilvusClient, you can flexibly configure connection parameters and the behavior of the gRPC channel pool:
- **uri** (str) — The address to connect to the Milvus server. For a local Docker container, http://localhost:19530 is typically used.
- **token** (str) — The authorization string formatted as username:password (defaults to "root:Milvus"). If cluster security is disabled via MILVUS_COMMON_SECURITY_AUTHORIZATIONENABLED=false (disabled by default), this parameter can be omitted.
- **db_name** (str) — The name of the specific database within Milvus that the client will interact with (defaults to "default").
- **timeout** (float) — The maximum wait time for a server response (timeout) for default network operations.
- **pool_size** (int) — The size of the connection pool. It specifies the maximum number of gRPC channels the client can keep open to concurrently handle competing asynchronous requests.
  
**Управление базой: Milvus Attu**

For convenient database administration in a production environment, Milvus Attu is used—an official graphical user interface that runs directly in a web browser. It serves the same role as pgAdmin does for PostgreSQL or Compass for MongoDB.

Through Attu, you can visually manage collections, create indexes, monitor the volume of loaded vectors, configure user permissions, and execute test search queries—all directly from your browser.

Instructions on how to deploy and work with the interface are available in the [Attu Quick Start Guide](https://milvus.io/docs/quickstart_with_attu.md).

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

### Search for similar videos

**Endpoint Response DTO for transmitting search results with relevance weights.**
```python
class ScoredVideoSearchResult(TypedDict):
    video_id: str
    score: float
```

**Vector Store Interface.**
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

**Service Layer for Semantic Search.**
```python
class SemanticService:
    def __init__(self, vector_repo: IVideoVectorRepository):
        self._vector_repo = vector_repo

    async def get_similar_videos(
        self,
        video_id: str,
        limit: int = 10,
        min_score: float = 0.75,
    ) -> list[ScoredVideoSearchResult]:

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

**A repository for working with Milvus data.**
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
    ) -> list[ScoredVideoSearchResult]:

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

        search_results: list[ScoredVideoSearchResult] = []

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

## Personalized Recommendations

### User Signals

User signals within the system include views, likes, dislikes, and comments. Each type of action is assigned a specific weight, which reflects the significance of that interaction in the content evaluation process.

A view serves as a foundational signal, recording the fact that a user has accessed the content. A like carries a positive weight, thereby boosting the overall score. A dislike carries a negative weight, lowering the content's rating. A comment carries the highest weight among all signals, as it requires active user engagement and reflects a more pronounced reaction.

### User Event Processing Pipeline

<img width="694" height="302" alt="image" src="https://github.com/user-attachments/assets/13e890d1-6e38-4350-9242-02547da7d050" />

****The Single-Queue Problem**

Using a single queue for all event types is a poor design choice due to the inability to scale effectively. View events arrive several times more frequently than dislike or comment events.

**What can be done about this?**

1. **Separation and Scaling of the Native Worker**

Write a worker script and scale it using Docker Compose with the command `docker-compose up --scale view-worker=5 -d`. Since RabbitMQ employs a Round-Robin algorithm, messages will be distributed and processed rapidly. Crucially, you must set the parameter `channel.basic_qos(prefetch_count=10)`. The worker will be solely responsible for writing data to the database—a very fast operation that is performed more efficiently in batches. Setting this value too low will result in high network latency. Conversely, omitting the parameter entirely or setting it too high may cause a sudden spike in CPU and memory load on the worker server.

2. **Using Celery.**

Segregate events into queues. Use the Celery framework atop RabbitMQ and execute tasks for each queue at varying intensities (number of parallel processes).

Launching a high-intensity worker for views:

```bash
celery -A tasks worker -Q views -c 10 --loglevel=info
```

Example of Celery configuration in Python code:

```python
from celery import Celery

app = Celery('tasks', broker='amqp://localhost')

app.conf.task_routes = {
    'tasks.process_view': {'queue': 'views'},
    'tasks.process_comment': {'queue': 'comments'},
}

@app.task
def process_view(data):
    return "View processed"

@app.task
def process_comment(data):
    return "Comment processed"
```

I also recommend configuring a connection pool in SQLAlchemy.

### ALS

ALS is a matrix factorization algorithm used in recommender systems to predict user preferences.
The essence of the method lies in decomposing the original matrix of user-item interactions into two matrices of lower dimensionality: a user profile matrix and an item feature matrix. The algorithm optimizes latent features by alternately fixing one matrix while training the other, thereby enabling the efficient processing of large-scale sparse data.

### User-Item Matrix

An interaction matrix in which rows represent users (user_id), columns represent videos (video_id), and the intersections represent a calculated engagement level (score: int). Since each user interacts with only a small fraction of the catalog, the matrix is ​​highly sparse.

If users A and B have rated several videos identically, the algorithm projects them to nearby points within a latent feature space. Consequently, if user A watches a new video and is satisfied with it, that content will, with a high degree of probability, be recommended to user B.

In the PostgreSQL database, this matrix is ​​mapped to the `user_video_interactions` table, which has the following structure:
- `user_id`
- `video_id`
- `score`
- `view_count`
- `likes_count`
- `dislikes_count`
- `comments_count`
- `first_interaction_at`
- `last_interaction_at`

If the weights change, they can be easily recalculated.

The project utilizes an aggregated approach: a single unique pair of user_id + video_id occupies exactly one row in the table. When a new event from a user arrives, it does not create a new record but updates the existing row in the database using the ON CONFLICT DO UPDATE clause. The ALS algorithm requires a prepared matrix as input, where each "user-video" pair has only one final numerical score. If we stored every action as a separate row, the database would have to execute heavy grouping and aggregation operations across millions of records during the nightly training process. On large volumes of data, such a query would take too long and generate massive CPU overhead. Aggregating at the write stage completely resolves this issue — the data is already stored in the exact format the model requires.

Frequent UPDATE operations in PostgreSQL lead to the accumulation of "dead" row versions due to MVCC, causing table and index bloat. To prevent performance degradation and ensure timely disk space reclamation, Autovacuum settings should be fine-tuned.

```SQL
ALTER TABLE user_video_interactions SET (
    autovacuum_enabled = true,
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_vacuum_threshold = 1000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 500
);
```

- **autovacuum_enabled = true:** Explicitly enables autovacuuming for this high-traffic table.
- **autovacuum_vacuum_scale_factor = 0.05:** The vacuuming process is triggered as soon as more than 5% of the total rows in the table have been updated or deleted (the default value for the entire DBMS is typically 20%, which is too slow for activity log tables).
- **autovacuum_vacuum_threshold = 1000:** The minimum absolute number of modified rows required to initiate vacuuming.
- **autovacuum_analyze_scale_factor = 0.02** and **autovacuum_analyze_threshold = 500:** Ensure frequent recalculation of data distribution statistics (triggered when 2% of rows change), which is essential for the PostgreSQL query planner to generate optimal execution plans for SELECT queries during nightly data imports.

**Development Prospects: Limiting the Time Window**

At the current stage of system operation, strict time constraints are not applied to requests, as the recommendation service was launched relatively recently. However, as the database grows—and to enhance the quality of recommendations—plans are in place to implement data filtering based on a sliding time window in the future.

A data slice covering the past four months will be added to the nightly export task:
```SQL
SELECT user_id, video_id, score
FROM user_video_interactions
WHERE last_interaction_at >= NOW() - INTERVAL '4 months';
```

The primary advantage is that the size of the matrix will cease to grow indefinitely; as a result, the model's nightly training will always proceed rapidly and will not overload the server's RAM. Furthermore, recommendations will become more accurate, as the algorithm will no longer take into account old, outdated views, focusing instead on what is of interest to the user right now.

The primary drawback is the complete loss of history. If a user has not accessed the platform for more than four months, the system forgets their preferences; upon returning, they are presented with empty recommendations, as if they were an entirely new user. In the future, to avoid this issue, plans are in place to implement a mechanism for gradually diminishing the significance of old data, rather than strictly deleting it.

### Model Training

Model training is initiated automatically during the night using Celery background tasks. When reading large volumes of data, an AccessShareLock is applied; while this does not prevent the simultaneous writing of new events to PostgreSQL, the execution of a resource-intensive SELECT query could lead to high CPU and disk I/O utilization.

1. **Data Extraction:** Raw pairs of identifiers and a calculated engagement score are retrieved from PostgreSQL.
2. **Type Conversion:** `user_id` and `video_id` are cast to the Pandas categorical type to generate the dense internal indices required by the algorithm.
3. **Sparse Matrix Creation:** A matrix in CSR (Compressed Sparse Row) format is constructed based on the calculated confidence levels. This optimizes RAM usage by eliminating the storage of zero elements.
   
```python
confidence = 1 + alpha * df["score"].astype(np.float32)
```

```python
matrix = csr_matrix((
        confidence,
        (
            df["user_id"].cat.codes,
            df["video_id"].cat.codes
        )
    ))
```

4. **Matrix Factorization:**

The `implicit` library factorizes the matrix into embeddings. The dimensionality of the latent factors (`factors`) is 384 (in accordance with the `settings.VECTOR_DIM` parameter), which ensures the correctness of subsequent nearest-neighbor searches.

**5. Vector Export:**

User vectors (`user_factors`) are saved to Redis in batches in a binary format.

Video content vectors (`item_factors`) are loaded into the Milvus vector database (the `item_behavioral` collection) to enable subsequent rapid search based on cosine distance.

### Obtaining Recommendations

```python
async def get_predictions(self, user_id: str, limit: int) -> list[str]:
        user_vec = await self.user_repo.get(user_id)
        if user_vec is None:
            return []

        search_results = await self.vector_storage.query_similar(
            collection_name=self._collection_name,
            vector=user_vec,
            limit=limit,
        )

        return [item["item_id"] for item in search_results]
```

## Cold Start

Cold start is a fairly common issue in the system, and here are the ways to resolve it:

### New Users

- **Displaying Popular and Trending Content**: To prevent a new user from seeing an empty page, the system initially determines their country or city via IP address and shows a selection of videos that have accumulated the highest number of views, likes, and comments in that region over the past 24 hours. This approach helps retain the user from the very first seconds using high-quality, proven content that is currently engaging the community around them.
- **Category Selection**: Upon the first launch of the app, the user is presented with a welcome screen featuring tiles of popular topics like "Sports", "Technology", or "Cooking" and is asked to select a few areas of interest. Based on this selection, the system instantly generates an initial interest profile, allowing it to immediately populate the feed with relevant videos instead of random content.
- **Mixed Sampling**: During the user's very first session, the algorithm builds a test feed consisting of top videos from completely different categories. The system closely monitors which videos the user stops to watch and which ones they skip, enabling it to understand real interests within just a few minutes of clicks and views, and begin narrowing down recommendations to their favorite topics.

### New Videos

- **Boosting Mechanism**: To prevent recently uploaded videos from getting lost among older content, the algorithm forcibly reserves exactly one-tenth (10%) of every user's feed for new arrivals. This guarantees that fresh videos receive their first few thousand impressions from real users, enabling the system to rapidly gather initial retention and click-through metrics to determine whether the video warrants further promotion.

### Popular Videos

- **Addressing the "Rich Get Richer" Phenomenon**: To stop older popular videos from permanently dominating the top rankings, a time-decay factor is integrated into the ranking algorithm. It operates by assigning baseline popularity scores to each video based on likes and views, which are then multiplied by a diminishing penalty coefficient that decreases with every day or hour since publication. Consequently, older videos gradually lose weight within the system, allowing new content to break into the top charts if it gains traction quickly.
  
- **Viral Content Isolation**: To protect individual user feeds from being flooded by identical viral clips, the system splits recommendations into two independent streams. Trending videos that currently capture the attention of the entire platform are forcibly routed to a separate tab or a dedicated interface block labeled "Trends." This isolates mass-appeal content, keeping the user's main home feed focused strictly on their unique hobbies and individual tastes rather than general public trends.

## Metrics

### CTR

This metric indicates how appealing a video's thumbnail and title are to users, and is measured as the ratio of clicks to impressions. If your video appeared on users' screens 100 times and was clicked by 5 people, the CTR would be 5%. A high CTR signals to the algorithm that the video's visual presentation and topic spark strong curiosity among the audience, and that the video is worth showing more frequently.

### Watch Time

This metric evaluates the actual quality of content and measures the total time users spend watching a specific video or using the application as a whole. Unlike a simple click (CTR), watch time cannot be gamed by a flashy thumbnail: if a video is boring or contains clickbait, the user will close it after just a couple of seconds. For a recommendation system, this serves as the primary signal that the content truly captures attention and delivers value.

### Recall@K

This metric evaluates an algorithm's ability to identify and recommend to a user precisely what they are theoretically interested in, specifically within the top *K* positions of the results list (for instance, among the top 10 or 20 videos). Simply put, if the system presents a user with a list of 10 videos—and that list includes at least a couple of clips that the individual ultimately enjoyed watching—it signifies that the algorithm has successfully fulfilled its task of initial content selection.

### NDCG

This metric evaluates not merely the presence of interesting videos within the recommendations, but also the correctness of their ordering, penalizing the system for poor sorting. The algorithm strives to ensure that the videos most relevant and potentially appealing to the user occupy the very top positions on the screen, while less interesting ones appear at the very bottom. If a video that is ideal for the user ends up in the tenth position instead of the first, the NDCG metric will drop noticeably, signaling the need to refine the ranking mechanism.

When a feed needs to be generated, the system retrieves the user's vector and mathematically compares it against the vectors of hundreds of relevant videos. The result of this comparison is a single numerical value—a `score`—for each video.

The algorithm then applies a standard sort, arranging the videos in descending order based on this `score`. Videos with the highest scores automatically occupy the first and second positions in the output array, while those with lower scores are relegated to the end of the list.
