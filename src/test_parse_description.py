from data_handler import merge_text_with_descriptions

# make temporary test function to debugging
import os

def test_parse_descriptions(texts_with_placeholders_output, output_image_descriptions_output, merged_output_test):

    merge_text_with_descriptions(texts_with_placeholders_output, output_image_descriptions_output, merged_output_test)


if __name__ == "__main__":

    # Example usage of the complete workflow
    # Adjust input_dir and output_dir as needed
    output_image_descriptions_dir = "./data/output/image_descriptions"
    texts_with_placeholders_dir = "./data/output/texts_with_placeholders"
    # list of txt in texts_with_placeholders_dir with path
    image_descriptions_output = [os.path.join(output_image_descriptions_dir, f) for f in os.listdir(output_image_descriptions_dir) if f.endswith("_descriptions.txt")]
    texts_with_placeholders_output = [os.path.join(texts_with_placeholders_dir, f) for f in os.listdir(texts_with_placeholders_dir) if f.endswith(".txt")]
    input_dir = "./data/pdfs"
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]

    merged_dir_test = os.path.join("./data/output", "final_merged_tests") # test directory experiment
    os.makedirs(merged_dir_test, exist_ok=True)
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(input_dir, pdf_file)
        pdf_name = os.path.splitext(pdf_file)[0]

        merged_output_test = os.path.join(merged_dir_test, f"{pdf_name}_final.txt")
        test_parse_descriptions(texts_with_placeholders_output[i-1], image_descriptions_output[i-1], merged_output_test)