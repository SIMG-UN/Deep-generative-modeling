# Deep Generative Modeling

Este repositorio contiene implementaciones y experimentos con modelos generativos profundos, enfocándose principalmente en modelos autoregresivos para procesamiento de lenguaje natural (NLP).

## 📋 Descripción del Proyecto

Este proyecto forma parte de un semillero de investigación dedicado al estudio y desarrollo de modelos generativos profundos. El enfoque principal está en:

- **Modelos Autoregresivos**: Implementación de arquitecturas que generan texto token por token
- **Procesamiento de Lenguaje Natural**: Técnicas fundamentales de NLP aplicadas a textos clásicos
- **Tokenización**: Desarrollo de tokenizadores personalizados para manejo de texto

## 🗂️ Estructura del Proyecto

```
Deep-generative-modeling/
├── Autoregressive Models/
│   ├── autoregresive_models.ipynb    # Implementación principal de modelos autoregresivos
│   └── basic_nlp.ipynb               # Fundamentos de NLP y preprocesamiento
├── Data/
│   ├── cervantes_2.txt               # Texto de Don Quijote de la Mancha
│   ├── cervantes.txt                 # Datos adicionales de Cervantes
│   └── shakespeare.txt               # Obras de Shakespeare
├── Raw_data/
│   └── Miguel de Cervantes El Ingenioso Hidalgo Don Quijote de la Mancha.pdf
├── requirements.txt                  # Dependencias del proyecto
└── README.md                         # Este archivo
```

## 🚀 Características Principales

### Modelos Autoregresivos
- **Tokenizador Personalizado**: Implementación de un tokenizador simple para procesamiento de texto
- **Vocabulario Dinámico**: Construcción automática de vocabulario con manejo de frecuencias mínimas
- **Tokens Especiales**: Soporte para tokens especiales como `<UNK>`, `<PAD>`, `<SOS>`, `<EOS>`

### Datos de Entrenamiento
- **Textos Clásicos**: Corpus basado en obras literarias clásicas en español e inglés
- **Don Quijote de la Mancha**: Texto completo de Miguel de Cervantes
- **Obras de Shakespeare**: Colección de textos shakespearianos

## 🛠️ Instalación y Configuración

### Prerrequisitos
- Python 3.8+
- Conda (recomendado para gestión de entornos)

### Instalación con Conda

1. **Clonar el repositorio**:
```bash
git clone <url-del-repositorio>
cd Deep-generative-modeling
```

2. **Crear entorno virtual con Conda**:
```bash
conda create -n deep-gen python=3.8
conda activate deep-gen
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

### Dependencias Principales

- **Deep Learning**: PyTorch (torch, torchvision, torchaudio)
- **NLP**: NLTK, spaCy, Transformers, Tokenizers
- **Análisis de Datos**: Pandas, NumPy, Scikit-learn
- **Visualización**: Matplotlib, Seaborn, WordCloud
- **Desarrollo**: Jupyter, IPywidgets

## 📚 Uso del Proyecto

### 1. Procesamiento Básico de NLP
```bash
jupyter notebook "Autoregressive Models/basic_nlp.ipynb"
```

### 2. Modelos Autoregresivos
```bash
jupyter notebook "Autoregressive Models/autoregresive_models.ipynb"
```

### Ejemplo de Uso del Tokenizador

```python
from tokenizer import SimpleTokenizer

# Inicializar tokenizador
tokenizer = SimpleTokenizer()

# Construir vocabulario
with open("Data/cervantes_2.txt", "r", encoding="utf-8") as file:
    text = file.read()

tokenizer.build_vocab(text, min_freq=2)

# Tokenizar texto
tokens = tokenizer.tokenize("En un lugar de la Mancha")
print(f"Tokens: {tokens}")
```

## 🔬 Componentes Técnicos

### Tokenización
- **Estrategia**: Tokenización a nivel de palabra
- **Preprocesamiento**: Normalización de texto y manejo de puntuación
- **Vocabulario**: Construcción automática con filtrado por frecuencia

### Arquitectura de Modelos
- **Redes Neuronales**: Implementación con PyTorch
- **Funciones de Activación**: Soporte para múltiples funciones de activación
- **Optimización**: Algoritmos de entrenamiento personalizables

## 📊 Datos y Corpus

### Textos Incluidos
- **Don Quijote de la Mancha**: Obra completa de Miguel de Cervantes
- **Shakespeare**: Colección de obras shakespearianas
- **Formato**: Archivos de texto plano (.txt) en UTF-8

### Preprocesamiento
- Limpieza de texto
- Normalización de caracteres
- Segmentación por tokens

## 🤝 Contribución

Este es un proyecto de investigación académica. Para contribuir:

1. Forkear el repositorio
2. Crear una rama para la nueva característica
3. Realizar commits con mensajes descriptivos
4. Enviar pull request con descripción detallada

## 📄 Licencia

Este proyecto es parte de un semillero de investigación académica. Consulte con los responsables del proyecto para información sobre uso y distribución.

## 📞 Contacto

Para preguntas sobre el proyecto o colaboraciones, contactar a través del semillero de investigación correspondiente.

## 🔗 Referencias

- **PyTorch Documentation**: https://pytorch.org/docs/
- **Transformers Library**: https://huggingface.co/docs/transformers/
- **NLTK**: https://www.nltk.org/

---

**Nota**: Este proyecto está en desarrollo activo como parte de actividades de investigación académica en modelos generativos profundos.
