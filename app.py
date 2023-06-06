import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
import itertools
from nltk.corpus import stopwords

# Initialize the image captioning processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)


def process_image(image, text):
    """
    Process the image and text inputs.

    Args:
        image (PIL.Image.Image): Input image.
        text (str): Additional text prompt.

    Returns:
        dict: Inputs for the caption generation model.
    """
    inputs = processor(image, text, return_tensors="pt")
    return inputs


def generate_sequences(inputs, n_beams, n_seqs=3):
    """
    Generate caption sequences using the model.

    Args:
        inputs (dict): Inputs for the caption generation model.
        n_beams (int): Number of beams for beam search.
        n_seqs (int, optional): Number of sequences to generate. Defaults to 3.

    Returns:
        list: Decoded caption sequences.
    """
    sequences = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=n_beams + 1,
        num_return_sequences=n_seqs,
        min_length=15,
        top_p=0.85,
    )
    decoded_seqs = []
    for seq in sequences:
        decoded_seqs.append(processor.decode(seq, skip_special_tokens=True))
    return decoded_seqs


def generate_hashtags(seqs):
    """
    Generate hashtags from caption sequences.

    Args:
        seqs (list): List of caption sequences.

    Returns:
        str: Generated hashtags.
    """
    seq = " ".join(seqs)
    caption_words = [
        word.lower() for word in seq.split(" ") if not word.startswith("#")
    ]

    # Remove stop words from captions
    stop_words = set(stopwords.words("english"))
    all_words = [word for word in caption_words if word not in stop_words]

    # Generate combinations of words for hashtags
    hashtags = []
    for n in range(1, 3):
        word_combinations = list(itertools.combinations(all_words, n))
        for combination in word_combinations:
            hashtag = "#" + "".join(combination)
            hashtags.append(hashtag)

    # Return top 10 hashtags by frequency
    top_hashtags = [
        tag
        for tag in sorted(set(hashtags), key=hashtags.count, reverse=True)
        if tag != "#"
    ]

    return ", ".join(top_hashtags[:10])


def process(image, text, n_seqs):
    """
    Process the image and generate captions and hashtags.

    Args:
        image (PIL.Image.Image): Input image.
        text (str): Additional text prompt.
        n_seqs (int): Number of captions to generate.

    Returns:
        tuple: Generated captions and hashtags.
    """
    inputs = process_image(image, text)
    seqs = generate_sequences(inputs, n_seqs, n_seqs)

    content = ""
    for i, seq in enumerate(seqs[:-1]):
        content += f"{i+1} : {seq}\n-----\n"

    hashtags = generate_hashtags(seqs)

    return content + f"{i+2} : {seqs[-1]}", hashtags


# Create the Gradio interface
with gr.Blocks(title="Image Captioning", theme="soft") as demo:
    gr.Markdown("An Image Captioning Generator")
    with gr.Row(variant="panel"):
        with gr.Column():
            image_input = gr.Image()
            text_input = gr.Text(
                label="Additional Prompt",
                placeholder="Enter text with which you want to generate the text",
            )
            n_seqs = gr.Slider(
                minimum=2,
                maximum=5,
                label="Number of Captions to Generate",
                value=3,
                step=1,
            )
        with gr.Column():
            result1 = gr.TextArea(lines=9, label="Generated Caption for the image")
            result2 = gr.TextArea(lines=4, label="Generated Hashtags")
    submit_btn = gr.Button("Generate Text")

    submit_btn.click(
        process,
        inputs=[image_input, text_input, n_seqs],
        outputs=[result1, result2],
        scroll_to_output=True,
    )
