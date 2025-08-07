import gradio as gr
from gradio_tabs.animation import animation
from gradio_tabs.vid_edit import vid_edit


def load_file(path):

	with open(path, 'r', encoding='utf-8') as f:
		content  = f.read()

	return content

custom_css = """
<style>
  body {
	font-family: Georgia, serif; /* Change to your desired font */
  }
  h1 {
	color: black; /* Change title color */
  }
</style>
"""


with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
	# ... (input/output setup remains unchanged)

	gr.HTML(load_file("assets/title.md"))
	with gr.Row():
		with gr.Accordion(open=False, label="Instruction"):
			gr.Markdown(load_file("assets/instruction.md"))
	
	with gr.Row():
		with gr.Tabs():
			animation()
			vid_edit()


if __name__ == "__main__":	
	
	demo.launch(
		server_name='0.0.0.0',
		#server_port=7803,
		server_port=10008,
		share=True,
		allowed_paths=[
			"./data/source",
			"./data/driving"]
	)
