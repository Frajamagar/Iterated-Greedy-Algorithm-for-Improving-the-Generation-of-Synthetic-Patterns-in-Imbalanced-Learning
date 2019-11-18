# An Iterated Greedy Algorithm for Improving the Generation of Synthetic Patterns in Imbalanced Learning

Creación de patrones sintéticos para conjuntos de datos desbalanceados mediante la metaheurística voraz iterativa.

Trabajo de Fin de Grado realizado para la universidad de Córdoba calificado con Matrícula de Honor.

El problema real que se plantea para este Trabajo Fin de Grado es, por una parte, la implementación de un algoritmo de remuestreo utilizando técnicas de over-sampling, inspirándose en los algoritmos existentes hasta la fecha, aplicándole además, algunas mejoras que se creen necesarias para hacerlo más eficiente o eficaz. Por otra parte, se implementará la metaheurística voraz iterativa para optimizar la creación de los patrones sintéticos, de forma que, se consiga generar un conjunto de sintéticos que aporte el mayor beneficio posible a la clasificación. En resumen, este Trabajo Fin de Grado, se divide en dos partes:

- **Algoritmo de remuestreo:** Se encargará de generar una serie de patrones sintéticos pertenecientes a la clase minoritaria, y que se unirán a la base de datos durante la fase de aprendizaje del algoritmo de clasificación , de tal forma, que se consiga un mejor resultado en el conjunto de test o al clasificar nuevas instancias. 

- **Metaheurística voraz iterativa:** Encargada de evolucionar el conjunto de sintéticos generado inicialmente hasta otro que ostente un mayor aporte de información al proceso de aprendizaje y clasificación. Esta metaheurística constará de una fase de destrucción aleatoria y parcial del conjunto de sintéticos y una reconstrucción guiada para llevar a cabo dicha evolución. Para asegurar la convergencia se utilizará un conjunto de validación. 


Por lo tanto, de la realización de este trabajo surgen dos nuevos algoritmos que pasan a formar parte cuerpo de metodologías y estrategias actuales para resolver este tipo de problemas. Estos algoritmos son:
- **ANESYN** (Adaptive neighbors Synthetic Sampling): Algoritmo de remuestreo diseñado para este proyecto sin ser optimizado mediante ninguna metaheurística
- **ANEIGSYN** (Adaptive neighbors Iterated Greedy Synthetic Sampling). Algoritmo completo que aplica tanto el algoritmo de remuestreo ANESYN, como la metaheurística voraz iterativa que lo optimiza.

Este trabajo fue presentado en el congreso internacional IWANN 2017 celebrado en Cádiz. https://link.springer.com/chapter/10.1007/978-3-319-59147-6_44

Para más información contactame o consulta los manuales disponobles en este repositorio.

Autor: Francisco Javier Maestre García
Email: frajamagar@gmail.com
