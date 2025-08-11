import markdown2
import weasyprint
import sys

def convert_markdown_to_pdf(input_file, output_file):
    # Read the markdown file
    with open(input_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(markdown_content)
    
    # Add basic CSS styling
    html_with_style = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
            }}
            h1 {{
                color: #333;
                border-bottom: 2px solid #333;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #666;
                margin-top: 30px;
            }}
            h3 {{
                color: #888;
                margin-top: 20px;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            pre {{
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                overflow-x: auto;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    pdf = weasyprint.HTML(string=html_with_style).write_pdf()
    
    # Save PDF to file
    with open(output_file, 'wb') as f:
        f.write(pdf)
    
    print(f"PDF successfully created: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_pdf.py <input_markdown_file> <output_pdf_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_markdown_to_pdf(input_file, output_file)
