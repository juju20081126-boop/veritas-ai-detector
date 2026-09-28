"""
Turnitin-Style PDF Report Generator
Produces an official, academic-grade "Originality & AI Authenticity Inspection Report"
using ReportLab with styled score badges, forensic metrics, and color-coded sentence annotations.
"""

import io
import time
from typing import Dict, Any, List

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch


def generate_ai_pdf_report(analysis_result: Dict[str, Any]) -> bytes:
    """
    Generates an official Turnitin-grade AI writing inspection report PDF.
    Returns the generated PDF as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b')
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )

    summary = analysis_result["summary"]
    metrics = analysis_result["metrics"]
    counts = analysis_result["counts"]
    sentences = analysis_result["sentences"]

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>VERITAS AI ORIGINALITY REPORT</b>", title_style),
            Paragraph(f"<b>Scan Date:</b> {time.strftime('%Y-%m-%d %H:%M:%S UTC')}<br/><b>Verification ID:</b> VER-{int(time.time()*1000)%10000000:07d}", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[3.8 * inch, 3.4 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceAfter=15))

    # 2. Executive Score Card
    ai_pct = summary["overall_ai_percentage"]
    if ai_pct >= 75:
        score_color = colors.HexColor('#dc2626')
        bg_card = colors.HexColor('#fef2f2')
        border_card = colors.HexColor('#f87171')
    elif ai_pct >= 50:
        score_color = colors.HexColor('#ea580c')
        bg_card = colors.HexColor('#fff7ed')
        border_card = colors.HexColor('#fb923c')
    elif ai_pct >= 25:
        score_color = colors.HexColor('#d97706')
        bg_card = colors.HexColor('#fffbeb')
        border_card = colors.HexColor('#fcd34d')
    else:
        score_color = colors.HexColor('#16a34a')
        bg_card = colors.HexColor('#f0fdf4')
        border_card = colors.HexColor('#86efac')

    score_badge_style = ParagraphStyle(
        'ScoreBadge',
        fontName='Helvetica-Bold',
        fontSize=36,
        leading=40,
        alignment=1,
        textColor=score_color
    )
    score_sub_style = ParagraphStyle(
        'ScoreSub',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        alignment=1,
        textColor=score_color
    )

    verdict_title_style = ParagraphStyle(
        'VerdictTitle',
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0f172a')
    )

    card_data = [
        [
            [
                Paragraph(f"{ai_pct}%", score_badge_style),
                Paragraph("OVERALL AI LIKELIHOOD", score_sub_style)
            ],
            [
                Paragraph(f"<b>Verdict:</b> {summary['verdict']}", verdict_title_style),
                Spacer(1, 4),
                Paragraph(summary['verdict_description'], body_style),
                Spacer(1, 6),
                Paragraph(
                    f"<b>Document:</b> {summary['filename']} &nbsp;|&nbsp; "
                    f"<b>Word Count:</b> {summary['word_count']} &nbsp;|&nbsp; "
                    f"<b>Sentences:</b> {summary['sentence_count']} &nbsp;|&nbsp; "
                    f"<b>Confidence:</b> {summary['confidence_percentage']}%",
                    subtitle_style
                )
            ]
        ]
    ]

    card_table = Table(card_data, colWidths=[2.2 * inch, 5.0 * inch])
    card_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_card),
        ('BOX', (0, 0), (-1, -1), 1, border_card),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 15))

    # 3. Forensic Metrics Matrix
    story.append(Paragraph("Forensic Linguistics & Model Telemetry", h2_style))

    lex = metrics["lexical_diversity"]
    read = metrics["readability"]
    syntax = metrics["syntax_variance"]

    metrics_data = [
        [
            Paragraph("<b>Metric Parameter</b>", table_header_style),
            Paragraph("<b>Measured Value</b>", table_header_style),
            Paragraph("<b>Benchmark Baseline</b>", table_header_style),
            Paragraph("<b>Diagnostic Interpretation</b>", table_header_style),
        ],
        [
            Paragraph("Mean Perplexity (PPL)", table_cell_style),
            Paragraph(f"{metrics['average_perplexity']}", table_cell_style),
            Paragraph("LLM: 12-25 | Human: 40-100+", table_cell_style),
            Paragraph("Lower values reflect machine predictability.", table_cell_style)
        ],
        [
            Paragraph("Burstiness Index (CV)", table_cell_style),
            Paragraph(f"{metrics['burstiness_index']} ({metrics['burstiness_label']})", table_cell_style),
            Paragraph("LLM: &lt; 0.25 | Human: &gt; 0.45", table_cell_style),
            Paragraph("Measures perplexity fluctuation across clauses.", table_cell_style)
        ],
        [
            Paragraph("Type-Token Ratio (TTR)", table_cell_style),
            Paragraph(f"{lex['ttr']} (Root: {lex['root_ttr']})", table_cell_style),
            Paragraph("Standard: 0.45 - 0.75", table_cell_style),
            Paragraph("Lexical diversity and vocabulary richness.", table_cell_style)
        ],
        [
            Paragraph("Sentence Uniformity", table_cell_style),
            Paragraph(f"{syntax['uniformity_score']} (CV: {syntax['cv_length']})", table_cell_style),
            Paragraph("Human standard: &gt; 0.45", table_cell_style),
            Paragraph("Robotic sentence rhythm indicator.", table_cell_style)
        ],
        [
            Paragraph("AI Cliché Density", table_cell_style),
            Paragraph(f"{metrics['total_ai_markers']} markers flagged", table_cell_style),
            Paragraph("Expected human: &lt; 1 per 300 words", table_cell_style),
            Paragraph("Characteristic LLM transitional signposts.", table_cell_style)
        ],
        [
            Paragraph("Readability Grade", table_cell_style),
            Paragraph(f"Grade {read['flesch_kincaid_grade']} (Ease: {read['flesch_reading_ease']})", table_cell_style),
            Paragraph("Collegiate: 10 - 14", table_cell_style),
            Paragraph("Flesch-Kincaid academic level index.", table_cell_style)
        ]
    ]

    metrics_table = Table(metrics_data, colWidths=[1.8 * inch, 1.8 * inch, 1.8 * inch, 1.8 * inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 15))

    # 4. Color Legend
    legend_text = (
        "<b>Legend:</b> &nbsp;&nbsp;"
        "<font color='#dc2626'><b>■ High AI Likelihood (&ge;75%)</b></font> &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<font color='#ea580c'><b>■ Likely AI (55-74%)</b></font> &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<font color='#d97706'><b>■ Mixed/Paraphrased (35-54%)</b></font> &nbsp;&nbsp;|&nbsp;&nbsp; "
        "<font color='#16a34a'><b>■ Likely Human (&lt;35%)</b></font>"
    )
    story.append(Paragraph(legend_text, subtitle_style))
    story.append(Spacer(1, 10))

    # 5. Annotated Full Document Inspection
    story.append(Paragraph("Annotated Sentence Heatmap Inspection", h2_style))

    doc_paragraphs = []
    current_para = []

    for s in sentences:
        s_prob = s["ai_percentage"]
        if s_prob >= 75:
            tag_color = "#fecaca"  # light red
            text_color = "#991b1b"
        elif s_prob >= 55:
            tag_color = "#fed7aa"  # light orange
            text_color = "#9a3412"
        elif s_prob >= 35:
            tag_color = "#fef08a"  # light yellow
            text_color = "#854d0e"
        else:
            tag_color = "#f1f5f9"  # clear / light gray
            text_color = "#334155"

        # Safe XML escaping
        escaped_sentence = (
            s["sentence"]
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
        )

        formatted_span = (
            f"<b>[{s['index']+1}]</b> "
            f"<font color='{text_color}'>{escaped_sentence}</font> "
        )
        current_para.append(formatted_span)

    full_annotated_text = "".join(current_para)
    annotated_style = ParagraphStyle(
        'AnnotatedDoc',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=14,
        textColor=colors.HexColor('#1e293b')
    )
    
    annotated_box = [
        [Paragraph(full_annotated_text, annotated_style)]
    ]
    annotated_table = Table(annotated_box, colWidths=[7.2 * inch])
    annotated_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e1')),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(annotated_table)
    story.append(Spacer(1, 15))

    # 6. Top Flagged Sentences Table (Deep Audit)
    story.append(Paragraph("Sentence-by-Sentence AI Probability Audit", h2_style))
    audit_data = [
        [
            Paragraph("<b>#</b>", table_header_style),
            Paragraph("<b>AI %</b>", table_header_style),
            Paragraph("<b>PPL</b>", table_header_style),
            Paragraph("<b>Sentence Excerpt</b>", table_header_style),
            Paragraph("<b>Forensic Evidence & Notes</b>", table_header_style)
        ]
    ]

    for s in sentences:
        s_prob = s["ai_percentage"]
        excerpt = s["sentence"]
        if len(excerpt) > 75:
            excerpt = excerpt[:72] + "..."
            
        escaped_excerpt = excerpt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        reason_text = "<br/>• ".join(s["reasons"])
        escaped_reasons = reason_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        if s_prob >= 75:
            badge_color = "#dc2626"
        elif s_prob >= 55:
            badge_color = "#ea580c"
        elif s_prob >= 35:
            badge_color = "#d97706"
        else:
            badge_color = "#16a34a"

        audit_data.append([
            Paragraph(f"{s['index']+1}", table_cell_style),
            Paragraph(f"<font color='{badge_color}'><b>{s_prob}%</b></font>", table_cell_style),
            Paragraph(f"{s['perplexity']:.1f}", table_cell_style),
            Paragraph(escaped_excerpt, table_cell_style),
            Paragraph(f"• {escaped_reasons}", table_cell_style)
        ])

    audit_table = Table(audit_data, colWidths=[0.4 * inch, 0.7 * inch, 0.6 * inch, 2.5 * inch, 3.0 * inch])
    audit_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('ALIGN', (0, 0), (2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(audit_table)

    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
