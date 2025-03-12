from docx import Document


class GenerateDocsReport():

    @classmethod
    def generateReport(cls, report_file, report):
        docx_file = report_file.replace(".html", ".docx")
        doc = Document()

        doc.add_heading(f"Отчёт \n {report}", 0)
        doc.save(docx_file)
