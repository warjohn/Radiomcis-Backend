import pkg_resources

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


class GeneratePdfReport():
    font_path = pkg_resources.resource_filename(
        'reportGeneration', 'fonts/Times_New_Roman.ttf'
    )
    print(font_path)
    pdfmetrics.registerFont(TTFont('TimesNewRoman', font_path))

    @classmethod
    def generatereport(cls, report_file, report):
        pdf_file = report_file.replace(".html", ".pdf")
        c = canvas.Canvas(pdf_file, pagesize=letter)
        c.setFont("TimesNewRoman", 12)
        c.drawString(100, 750, f"Отчёт \n {report}")
        y_position = 690
        c.save()