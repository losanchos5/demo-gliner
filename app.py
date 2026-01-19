import sys
import os

# Configurar codificación UTF-8 para Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    os.environ["PYTHONIOENCODING"] = "utf-8"

import gradio as gr
from gliner2 import GLiNER2

# Cargar modelo GLiNER2 Large
print("Cargando modelo GLiNER2 Large...")
modelo = GLiNER2.from_pretrained("fastino/gliner2-large-v1")
print("Modelo cargado!")

# Mapeo global de entidades para consistencia
mapeo_entidades = {}


def parsear_categorias(categorias_texto):
    """Convierte string de categorías separadas por coma en lista."""
    if not categorias_texto.strip():
        return ["persona", "empresa", "ubicacion", "fecha", "email", "telefono"]
    return [cat.strip().lower() for cat in categorias_texto.split(",") if cat.strip()]


def pseudonimizar(texto, categorias):
    """Extrae entidades y las reemplaza con placeholders."""
    global mapeo_entidades

    if not texto.strip():
        return texto

    # Extraer entidades
    resultado = modelo.extract_entities(texto, categorias)
    texto_pseudo = texto

    # Procesar cada categoría de entidades encontradas
    if "entities" in resultado:
        for categoria, valores in resultado["entities"].items():
            for valor in valores:
                if valor not in mapeo_entidades:
                    # Contar cuántas entidades de esta categoría ya tenemos
                    contador = sum(1 for v in mapeo_entidades.values()
                                   if v.startswith(f"[{categoria.upper()}_"))
                    mapeo_entidades[valor] = f"[{categoria.upper()}_{contador + 1}]"
                # Reemplazar en el texto
                texto_pseudo = texto_pseudo.replace(valor, mapeo_entidades[valor])

    return texto_pseudo


def procesar_mensaje(mensaje, categorias_texto, historial_original, historial_pseudo):
    """Procesa un mensaje y actualiza ambos historiales."""
    if not mensaje.strip():
        return historial_original, historial_pseudo, ""

    categorias = parsear_categorias(categorias_texto)
    mensaje_pseudo = pseudonimizar(mensaje, categorias)

    # Agregar a historiales
    historial_original = historial_original + [[mensaje, None]]
    historial_pseudo = historial_pseudo + [[mensaje_pseudo, None]]

    return historial_original, historial_pseudo, ""


def limpiar_chat():
    """Limpia los historiales y el mapeo de entidades."""
    global mapeo_entidades
    mapeo_entidades = {}
    return [], [], ""


# Interfaz Gradio
with gr.Blocks(title="Demo GLiNER2 - Pseudonimizador") as demo:
    gr.Markdown("# Demo GLiNER2 - Pseudonimizador de Texto")
    gr.Markdown("Escribe mensajes en el chat izquierdo y ve la versión pseudonimizada a la derecha en tiempo real.")

    # Campo de categorías
    categorias_input = gr.Textbox(
        label="Categorías de entidades a detectar (separadas por coma)",
        placeholder="persona, empresa, ubicacion, fecha, email, telefono, direccion",
        value="persona, empresa, ubicacion, fecha, email, telefono",
        lines=1
    )

    # Dos columnas para los chats
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Chat Original")
            chat_original = gr.Chatbot(
                label="Mensajes originales",
                height=400,
                value=[]
            )

        with gr.Column():
            gr.Markdown("### Texto Pseudonimizado")
            chat_pseudo = gr.Chatbot(
                label="Mensajes pseudonimizados",
                height=400,
                value=[]
            )

    # Input del usuario
    with gr.Row():
        mensaje_input = gr.Textbox(
            label="Escribe tu mensaje",
            placeholder="Ej: Hola, soy María García y trabajo en Google desde Madrid",
            lines=2,
            scale=4
        )
        enviar_btn = gr.Button("Enviar", variant="primary", scale=1)

    limpiar_btn = gr.Button("Limpiar conversación")

    # Función para procesar con el formato correcto de mensajes
    def procesar_y_formatear(mensaje, categorias_texto, hist_orig, hist_pseudo):
        if not mensaje.strip():
            return hist_orig, hist_pseudo, ""

        categorias = parsear_categorias(categorias_texto)
        mensaje_pseudo = pseudonimizar(mensaje, categorias)

        # Formato de mensajes: diccionarios con role y content
        hist_orig = hist_orig + [{"role": "user", "content": mensaje}]
        hist_pseudo = hist_pseudo + [{"role": "user", "content": mensaje_pseudo}]

        return hist_orig, hist_pseudo, ""

    def limpiar():
        global mapeo_entidades
        mapeo_entidades = {}
        return [], [], ""

    # Eventos
    enviar_btn.click(
        fn=procesar_y_formatear,
        inputs=[mensaje_input, categorias_input, chat_original, chat_pseudo],
        outputs=[chat_original, chat_pseudo, mensaje_input]
    )

    mensaje_input.submit(
        fn=procesar_y_formatear,
        inputs=[mensaje_input, categorias_input, chat_original, chat_pseudo],
        outputs=[chat_original, chat_pseudo, mensaje_input]
    )

    limpiar_btn.click(
        fn=limpiar,
        outputs=[chat_original, chat_pseudo, mensaje_input]
    )


if __name__ == "__main__":
    demo.launch()
