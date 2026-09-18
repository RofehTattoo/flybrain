# FlyBrain V1.04

## Objetivo
V1.04 mantiene exactamente 16.669 neuronas del MaleCNS v1.0 y da prioridad explícita a la preservación de rutas funcionales medidas en el conectoma reducido. La selección ya no depende únicamente de grado/superclase: se conservan preferentemente neuronas intermedias que participan en rutas de dos saltos `sensor -> célula -> DN` y `DN -> célula -> MN`, separando familias de avance, orientación/giro, escape y sensorimotora general.

El movimiento del cuerpo virtual sigue procediendo exclusivamente de actividad medida de neuronas motoras VNC retenidas. Las etiquetas de DN/MN y las puntuaciones de ruta son metadatos derivados del conectoma y no crean conexiones artificiales.

## Fuente científica
MaleCNS v1.0 — Janelia/FlyEM. Dataset CC-BY-4.0.
https://male-cns.janelia.org/download/

Berg et al., *Cell* (2026), “Sexual dimorphism in the complete connectome of the Drosophila male central nervous system”, DOI 10.1016/j.cell.2026.08.015.

## Arquitectura V1.04

### Selección connectómica
- 16.669 neuronas exactamente.
- Todos los `descending_neuron` y `vnc_motor` anotados se conservan cuando forman parte del censo trazado.
- Se conserva diversidad de tipos publicados.
- Se añade selección por rutas de dos saltos medidas en el grafo publicado.
- Se priorizan rutas sensorial→forward, visual→turn, visual/mecano→escape y DN→intermedia→MN.
- Las aristas embebidas son exclusivamente aristas publicadas entre neuronas retenidas.

### Metadatos de rutas
El formato FBC102 añade tres valores `Float` por neurona:
- `routeForward`
- `routeTurn`
- `routeEscape`

Son puntuaciones topológicas normalizadas calculadas a partir de rutas de dos saltos del grafo completo. Se utilizan para selección y diagnóstico; no modifican la conectividad.

### Dinámica neuronal
- LIF con paso neural fijo de 20 ms (50 Hz).
- Traza sináptica de corta duración (~45 ms).
- Sin plasticidad experimental por defecto.
- El límite de corriente sináptica se aplica después de la ganancia poblacional para evitar la saturación prematura que se observó en V1.01.
- El estado locomotor interno sigue siendo modulador y no escribe directamente posición, velocidad ni rumbo.

### Sensores
- Visual: codificación espacial y direccional.
- Olfativo: concentración con asimetría bilateral.
- Gustativo: contacto/proximidad muy cercana.
- Mecanosensorial: amenaza/contacto y feedback de límites.
- PELIGRO incorpora una componente visual de expansión lenta (looming) y una componente mecanosensorial; ninguna de ellas escribe una orden motora.

### Selección de acción
La aproximación ya no equivale a “caminar”: necesita evidencia neural de avance y contexto olfativo/gustativo.

El escape utiliza conjuntamente:
- actividad de DN etiquetados como escape;
- actividad de células intermedias con puntuación de ruta de escape;
- actividad motora defensiva medida.

El salto deja de ser la definición exclusiva de escape.

## Integridad
El binario FBC102 tiene:
- cabecera de 8 bytes;
- 16 bytes de cabecera total;
- registros de nodo de 25 bytes;
- registros de arista de 12 bytes;
- validación de tamaño exacto;
- validación de todos los índices;
- validación de todos los pesos finitos y no nulos;
- validación de los metadatos de ruta.

Si el connectome falla al cargar, la aplicación entra en `fail-closed`: no se crea una red neuronal alternativa.

## Limitaciones científicas
La reducción a 16.669 neuronas es un subconjunto del MaleCNS completo, no un décimo espacial exacto de cada circuito. Las puntuaciones de ruta son una herramienta de preservación topológica, no una afirmación de que cada neurona tenga una función conductual única. Los roles DN usan nombres de tipos publicados cuando existe evidencia funcional establecida (por ejemplo DNg100/DNg97/DNb08 para marcha y DNp01/Giant Fiber para escape). La selección de escape protege también las entradas visuales hacia el DNp01 (LC4/LPLC2) cuando están presentes en las anotaciones y aristas retenidas. La dinámica LIF, el signo neurotransmisor-resuelto y la mecánica corporal siguen siendo aproximaciones computacionales.

En particular, el glutamato se modela como inhibitorio en esta reducción siguiendo la convención empleada en simulaciones recientes del VNC de Drosophila; no se representan excepciones dependientes del receptor.


## Interfaz V1.04
La interfaz se simplifica deliberadamente para que la simulación sea lo primero que se vea:
- 16.669 neuronas y referencia `MaleCNS v1.0`.
- Solo se muestran comidas, escapes, saciedad, memoria y FPS en el bloque de estado.
- El panel inferior combina una silueta 2D simplificada del SNC de la mosca con una selección pequeña de neuronas realmente retenidas.
- Las líneas del mapa son conexiones reales entre las neuronas representativas elegidas; no son una red decorativa inventada.
- Las alas tienen una animación visual de batido ligada a la actividad motora de alas medida y al desplazamiento.
- Se incluye un zumbido de mosca generado como recurso local. El audio se activa suavemente durante el movimiento o una interacción sensorial y se detiene en reposo.
- La animación y el audio son capas de presentación: no modifican la dinámica neuronal ni crean órdenes motoras.

La versión Android es `1.04` / `versionCode 104`; el formato binario interno continúa siendo `FBC102` para no romper la compatibilidad del lector existente.
