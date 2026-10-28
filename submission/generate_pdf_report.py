"""
Academic PDF Report Generator for Nassau Intelligence
Converts PROJECT_REPORT.md into a professional, printable academic PDF.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def build_pdf():
    base_dir = Path(__file__).resolve().parent
    report_md_path = base_dir / "PROJECT_REPORT.md"
    pdf_out_path = base_dir / "PROJECT_REPORT.pdf"
    
    doc = SimpleDocTemplate(
        str(pdf_out_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Academic Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0A1D13'),
        alignment=1, # Center
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#087F5B'),
        alignment=1, # Center
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#0A1D13'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#112F20'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-8,
        spaceAfter=3
    )
    
    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#0F172A'),
        backColor=colors.HexColor('#F1F5F9'),
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6
    )
    
    story = []
    
    story.append(Paragraph("NASSAU INTELLIGENCE: FACTORY REALLOCATION & SHIPPING OPTIMIZATION SYSTEM", title_style))
    story.append(Paragraph("An Econometrically Audited, Counterfactual Decision-Support Platform for Multi-Echelon Supply Chains", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0A1D13'), spaceAfter=12))
    
    with open(report_md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_code_block = False
    code_lines = []
    in_table = False
    table_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Skip top markdown header since we already added title
        if stripped.startswith("# ACADEMIC PROJECT REPORT") or stripped.startswith("## NASSAU INTELLIGENCE") or stripped.startswith("### An Econometrically"):
            continue
        if stripped == "---":
            continue
            
        # Code blocks
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "<br/>".join([c.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;") for c in code_lines])
                story.append(Paragraph(code_text, code_style))
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            continue
            
        if in_code_block:
            code_lines.append(line.rstrip())
            continue
            
        # Tables
        if stripped.startswith("|") and stripped.endswith("|"):
            if "---" in stripped and not in_table:
                continue
            table_lines.append(stripped)
            in_table = True
            continue
        elif in_table:
            # Process buffered table
            in_table = False
            raw_rows = []
            for tline in table_lines:
                if ":---" in tline or "---:" in tline or "---" in tline:
                    continue
                cells = [c.strip() for c in tline.split("|")[1:-1]]
                if cells:
                    raw_rows.append(cells)
            
            if raw_rows:
                # Wrap table cells in Paragraphs
                flowable_data = []
                for row_idx, row in enumerate(raw_rows):
                    row_data = []
                    for col in row:
                        p_font = 'Helvetica-Bold' if row_idx == 0 else 'Helvetica'
                        p_size = 7.5
                        p_color = colors.HexColor('#0A1D13') if row_idx == 0 else colors.HexColor('#1E293B')
                        cell_style = ParagraphStyle(
                            f'Cell_{row_idx}', fontName=p_font, fontSize=p_size, leading=10, textColor=p_color
                        )
                        row_data.append(Paragraph(col.replace("**", ""), cell_style))
                    flowable_data.append(row_data)
                
                # Determine table column widths
                col_count = len(flowable_data[0])
                table_w = 500 / col_count
                t = Table(flowable_data, colWidths=[table_w]*col_count)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0A1D13')),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
                    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94A3B8')),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 6))
            table_lines = []
            
        # Empty line
        if not stripped:
            continue
            
        # Headings
        if stripped.startswith("## "):
            story.append(Paragraph(stripped[3:], h1_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=4))
        elif stripped.startswith("### "):
            story.append(Paragraph(stripped[4:], h2_style))
        elif stripped.startswith("#### "):
            story.append(Paragraph(stripped[5:], h2_style))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            content = stripped[2:].replace("**", "<b>", 1)
            content = content.replace("**", "</b>", 1)
            story.append(Paragraph(f"&bull; {content}", bullet_style))
        elif stripped.startswith("1. ") or stripped.startswith("2. ") or stripped.startswith("3. ") or stripped.startswith("4. ") or stripped.startswith("5. ") or stripped.startswith("6. "):
            num, rest = stripped.split(". ", 1)
            rest = rest.replace("**", "<b>", 1).replace("**", "</b>", 1)
            story.append(Paragraph(f"<b>{num}.</b> {rest}", bullet_style))
        else:
            p_text = stripped.replace("**", "<b>", 1).replace("**", "</b>", 1)
            story.append(Paragraph(p_text, body_style))
            
    doc.build(story)
    print(f"PDF successfully built at {pdf_out_path}")

if __name__ == "__main__":
    build_pdf()
