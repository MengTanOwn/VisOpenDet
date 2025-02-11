# import gradio as gr
# from gradio_image_prompter import ImagePrompter

# demo = gr.Interface(
#     lambda prompts: (prompts["image"], prompts["points"]),
#     ImagePrompter(show_label=False),
#     [gr.Image(show_label=False), gr.Dataframe(label="Points")],
# )
# demo.launch()
score_T = 0.2
prompts_number =2
result_data_root = f'testdata/Dataset/Query_out_{score_T}/propmt_{prompts_number}'
# catelist = list(range(0,26))
print(result_data_root)