"""
=============================================================================
Categorized Technical Interview Repository
Intelligent Multimodal Mock Interview Assessment System
=============================================================================
Provides categorized questions and benchmark reference answers across:
  - Python
  - C++
  - Java
  - DSA (Data Structures & Algorithms)
  - DBMS (Database Management Systems)
  - OS (Operating Systems)
  - Computer Networks
  - Machine Learning (ML)
=============================================================================
"""

CATEGORIZED_QUESTIONS = {
    "Machine Learning": [
        {
            "id": "ML_01",
            "question": "What is the difference between supervised and unsupervised learning?",
            "reference_answer": (
                "Supervised learning trains algorithms on labeled datasets where each training example "
                "has an associated target label, enabling the model to learn a predictive mapping from inputs to outputs "
                "(such as classification and regression). In contrast, unsupervised learning discovers inherent patterns, "
                "groupings, or probability distributions from unlabeled data without ground-truth supervisory signals, "
                "commonly applied in clustering (e.g., K-Means), dimensionality reduction (e.g., PCA), and anomaly detection."
            ),
            "key_concepts": ["labeled data", "unlabeled data", "target label", "classification", "regression", "clustering", "patterns"]
        },
        {
            "id": "ML_02",
            "question": "What is the bias-variance tradeoff in machine learning?",
            "reference_answer": (
                "The bias-variance tradeoff describes the tension between two sources of error in predictive models. "
                "High bias occurs when a model is overly simplistic and makes strong assumptions (underfitting), "
                "failing to capture underlying patterns. High variance occurs when a model is overly complex and "
                "memorizes training noise (overfitting), generalizing poorly to unseen data. Optimal model performance "
                "minimizes total expected generalization error by balancing model complexity through regularization, "
                "cross-validation, and ensemble methods."
            ),
            "key_concepts": ["bias", "variance", "underfitting", "overfitting", "generalization", "complexity", "regularization"]
        },
        {
            "id": "ML_03",
            "question": "How does Gradient Boosting differ from Random Forest?",
            "reference_answer": (
                "Random Forest is a bagging ensemble that trains multiple independent decision trees in parallel on bootstrap "
                "samples and aggregates their predictions via majority voting or averaging to reduce variance. Gradient Boosting "
                "is a sequential boosting algorithm where each new decision tree is trained to predict and minimize the pseudo-residuals "
                "(gradient of the loss function) of the preceding trees, systematically reducing bias."
            ),
            "key_concepts": ["bagging", "boosting", "parallel", "sequential", "residuals", "variance reduction", "bias reduction"]
        }
    ],
    "Python": [
        {
            "id": "PY_01",
            "question": "What is the Global Interpreter Lock (GIL) in Python and how does it impact multi-threading?",
            "reference_answer": (
                "The Global Interpreter Lock (GIL) is a mutex in CPython that prevents multiple native threads from executing "
                "Python bytecode simultaneously. Its purpose is to simplify memory management and protect reference counts from "
                "race conditions. Consequently, CPU-bound multi-threaded Python programs do not achieve true parallelism across "
                "multiple cores; multi-processing or asynchronous I/O is required for parallel execution."
            ),
            "key_concepts": ["GIL", "CPython", "mutex", "bytecode", "multi-threading", "multi-processing", "CPU-bound", "concurrency"]
        },
        {
            "id": "PY_02",
            "question": "Explain the difference between mutable and immutable data types in Python with examples.",
            "reference_answer": (
                "Mutable objects in Python can have their internal state or values modified in-place after creation without changing "
                "their memory address (id). Examples include lists, dictionaries, and sets. Immutable objects cannot be modified after "
                "creation; any update creates a new object in memory. Examples include integers, floats, strings, and tuples."
            ),
            "key_concepts": ["mutable", "immutable", "in-place", "memory address", "list", "dict", "tuple", "string"]
        },
        {
            "id": "PY_03",
            "question": "What are generators in Python, and how does the yield keyword operate?",
            "reference_answer": (
                "Generators are iterators defined as functions using the yield keyword rather than return. When yield is executed, "
                "the function's execution state, local variables, and instruction pointer are suspended and returned to the caller, "
                "allowing lazy evaluation. This produces values one at a time on-demand, optimizing memory consumption for large datasets."
            ),
            "key_concepts": ["generator", "yield", "lazy evaluation", "iterator", "memory efficiency", "state suspension"]
        }
    ],
    "DSA": [
        {
            "id": "DSA_01",
            "question": "What is the difference between an Array and a Linked List in terms of memory and time complexity?",
            "reference_answer": (
                "Arrays allocate contiguous blocks of memory, offering O(1) random access by index, but require O(N) time for insertions "
                "or deletions at arbitrary positions due to element shifting, with fixed or amortized reallocation costs. Linked Lists allocate "
                "nodes non-contiguously with pointer overhead, providing O(1) insertions or deletions when a pointer is already available, "
                "but require O(N) sequential traversal time for random lookups."
            ),
            "key_concepts": ["contiguous memory", "pointers", "O(1) access", "O(N) traversal", "insertion", "cache locality"]
        },
        {
            "id": "DSA_02",
            "question": "Explain the difference between QuickSort and MergeSort, including their time and space complexities.",
            "reference_answer": (
                "MergeSort is a divide-and-conquer, stable sorting algorithm with guaranteed O(N log N) time complexity across best, "
                "worst, and average cases, requiring O(N) auxiliary memory for merging. QuickSort is an in-place divide-and-conquer "
                "algorithm that partitions elements around a pivot. It achieves O(N log N) average time and O(log N) stack space, "
                "but degrades to O(N^2) worst-case time if poorly chosen pivots are selected."
            ),
            "key_concepts": ["divide-and-conquer", "stable", "pivot", "in-place", "O(N log N)", "O(N^2)", "auxiliary space"]
        },
        {
            "id": "DSA_03",
            "question": "What is a Hash Table and how are hash collisions handled?",
            "reference_answer": (
                "A Hash Table maps keys to bucket indices using a hash function, targeting average O(1) time complexity for lookup, "
                "insertion, and deletion. Hash collisions occur when different keys produce identical bucket indices. Common resolution "
                "techniques include Separate Chaining (each bucket contains a linked list or balanced tree) and Open Addressing "
                "(linear probing, quadratic probing, or double hashing to find alternative vacant slots)."
            ),
            "key_concepts": ["hash function", "collision", "separate chaining", "open addressing", "O(1) average", "probing"]
        }
    ],
    "DBMS": [
        {
            "id": "DBMS_01",
            "question": "What are ACID properties in a relational database management system?",
            "reference_answer": (
                "ACID properties guarantee transaction reliability in database systems: Atomicity ensures all operations succeed or "
                "the entire transaction is rolled back; Consistency guarantees transitions between valid schema-compliant states; "
                "Isolation ensures concurrent transactions execute without interfering with one another; and Durability guarantees "
                "committed transactions survive system crashes through write-ahead logging."
            ),
            "key_concepts": ["Atomicity", "Consistency", "Isolation", "Durability", "transaction", "rollback", "write-ahead logging"]
        },
        {
            "id": "DBMS_02",
            "question": "Explain the difference between clustered and non-clustered indexing in SQL.",
            "reference_answer": (
                "A clustered index defines the physical storage order of table data rows on disk, meaning a table can have only one "
                "clustered index (typically the primary key), with leaf nodes containing actual data pages. A non-clustered index is a "
                "separate B-tree structure where leaf nodes contain index keys and row pointers (RID or clustered key) pointing back to "
                "the actual data records."
            ),
            "key_concepts": ["clustered index", "non-clustered index", "physical storage", "B-tree", "pointers", "leaf nodes"]
        }
    ],
    "OS": [
        {
            "id": "OS_01",
            "question": "What is the difference between a Process and a Thread in modern operating systems?",
            "reference_answer": (
                "A process is an independent execution unit with its own dedicated virtual address space, memory pages, file descriptors, "
                "and environment, isolated by the OS kernel. A thread is a lightweight execution stream within a process that shares "
                "the parent process's memory space, code section, and data segments, but maintains its own program counter, registers, "
                "and call stack, enabling low-overhead context switching."
            ),
            "key_concepts": ["address space", "virtual memory", "lightweight", "context switch", "shared memory", "program counter", "isolation"]
        },
        {
            "id": "OS_02",
            "question": "What is Deadlock and what are the four necessary Coffman conditions for it to occur?",
            "reference_answer": (
                "A deadlock is a situation where a set of concurrent processes are permanently blocked because each process holds a resource "
                "and waits for another resource held by another process. The four necessary Coffman conditions are: Mutual Exclusion (non-shareable "
                "resources), Hold and Wait (processes hold allocated resources while requesting new ones), No Preemption (resources cannot be "
                "forcibly reclaimed), and Circular Wait (a closed chain of processes waiting on cyclical dependencies)."
            ),
            "key_concepts": ["deadlock", "mutual exclusion", "hold and wait", "no preemption", "circular wait", "Coffman conditions"]
        }
    ],
    "Computer Networks": [
        {
            "id": "CN_01",
            "question": "Explain the differences between TCP and UDP protocols and when to use each.",
            "reference_answer": (
                "TCP (Transmission Control Protocol) is connection-oriented, reliable, and byte-stream oriented, providing guaranteed delivery, "
                "in-order packet sequencing, flow control, and congestion control via a three-way handshake, suited for web browsing (HTTP), "
                "email, and file transfers. UDP (User Datagram Protocol) is connectionless, unreliable, and lightweight, transmitting datagrams "
                "with minimal header overhead and lower latency, suited for real-time applications such as video streaming, VoIP, and online gaming."
            ),
            "key_concepts": ["connection-oriented", "connectionless", "three-way handshake", "reliability", "flow control", "latency", "UDP", "TCP"]
        },
        {
            "id": "CN_02",
            "question": "What happens under the hood when a user types a URL into a web browser and hits Enter?",
            "reference_answer": (
                "The browser resolves the domain name into an IP address via DNS lookup (checking browser cache, OS cache, router, and recursive "
                "DNS servers). It initiates a TCP three-way handshake with the server's IP address on port 80/443, followed by a TLS handshake for "
                "HTTPS encryption. The browser issues an HTTP GET request, the server responds with status codes and resources (HTML/CSS/JS), and "
                "the browser's rendering engine constructs the DOM, CSSOM, and render tree to display the page."
            ),
            "key_concepts": ["DNS resolution", "TCP handshake", "TLS encryption", "HTTP GET", "DOM tree", "render engine"]
        }
    ],
    "Java": [
        {
            "id": "JV_01",
            "question": "What is the difference between an abstract class and an interface in Java 8+?",
            "reference_answer": (
                "An abstract class can maintain instance state (fields), constructors, and both abstract and non-abstract concrete methods, "
                "supporting single class inheritance. An interface defines a behavioral contract and supports multiple inheritance of type. "
                "Starting with Java 8, interfaces can contain default and static methods with implementation bodies, and Java 9 added private methods, "
                "but interfaces still cannot maintain mutable instance state."
            ),
            "key_concepts": ["abstract class", "interface", "multiple inheritance", "default methods", "instance state", "contract"]
        },
        {
            "id": "JV_02",
            "question": "Explain how the Java Garbage Collector manages heap memory and the generational hypothesis.",
            "reference_answer": (
                "The Java Garbage Collector automatically reclaims unreachable heap objects based on the generational hypothesis, which posits "
                "that most objects die young. Heap memory is partitioned into Young Generation (Eden and Survivor spaces: S0, S1) and Old "
                "(Tenured) Generation. New objects are allocated in Eden; survivors of Minor GC passes are promoted to Tenured space, which is "
                "subsequently collected by Major/Full GC algorithms."
            ),
            "key_concepts": ["garbage collection", "generational hypothesis", "Eden space", "Survivor spaces", "Tenured", "heap memory"]
        }
    ],
    "C++": [
        {
            "id": "CPP_01",
            "question": "Explain the concept of RAII (Resource Acquisition Is Initialization) and smart pointers in modern C++.",
            "reference_answer": (
                "RAII binds the lifecycle of a resource (heap memory, file handles, mutex locks) to the object lifetime of a stack-allocated variable. "
                "The resource is acquired in the constructor and deterministically released in the destructor when exiting scope, preventing leaks "
                "even during exceptions. Modern C++ implements RAII through smart pointers: std::unique_ptr (exclusive single ownership) and "
                "std::shared_ptr with std::weak_ptr (reference-counted shared ownership)."
            ),
            "key_concepts": ["RAII", "destructor", "smart pointers", "unique_ptr", "shared_ptr", "resource leak", "exception safety"]
        },
        {
            "id": "CPP_02",
            "question": "What are virtual functions and vtables in C++, and how do they enable runtime polymorphism?",
            "reference_answer": (
                "Virtual functions in C++ allow derived classes to override base class implementations for dynamic runtime dispatch. When a class declares "
                "a virtual function, the compiler creates a virtual method table (vtable) containing function pointers for that class, and inserts a hidden "
                "pointer (vptr) into each object instance. Calling a virtual method resolves the address through the object's vptr at runtime."
            ),
            "key_concepts": ["virtual function", "vtable", "vptr", "dynamic dispatch", "polymorphism", "runtime resolution"]
        }
    ]
}

def get_categories():
    return list(CATEGORIZED_QUESTIONS.keys())

def get_questions_by_category(category: str):
    return CATEGORIZED_QUESTIONS.get(category, [])

