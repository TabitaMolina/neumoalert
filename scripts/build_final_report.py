"""
Construye el Informe Final oficial del Capstone SIC 2026 IA para el Equipo 2 (NEUMOALERT),
poblando una copia de la plantilla oficial de Samsung (SIC_IA_Capstone_Informe_Final.docx)
con portada, redacción técnica completa (Secciones 1 a 5), tablas formateadas,
5 figuras de alta resolución y marcadores visibles para los datos finales de la CNN.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

BASE_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = BASE_DIR / "reports" / "figures"
OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TEMPLATE_PATH = BASE_DIR / "reports" / "SIC_IA_Capstone_Informe_Final.docx"
OUTPUT_PATH = OUTPUT_DIR / "NEUMOALERT_Informe_Final_SIC2026.docx"

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MC_NS = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"

# Paleta sobria clínica/corporativa alineada al azul Samsung
COLOR_PRIMARY = RGBColor(0x0B, 0x2D, 0x48)   # Azul oscuro títulos/subtítulos
COLOR_ACCENT = RGBColor(0x0F, 0x4C, 0x75)    # Azul médico tablas/énfasis
COLOR_BODY = RGBColor(0x1E, 0x29, 0x3B)      # Gris pizarra oscuro cuerpo
COLOR_MUTED = RGBColor(0x47, 0x55, 0x69)     # Gris pies de figura
FONT_NAME = "Calibri"


def get_xml_text(el) -> str:
    """Extrae únicamente el texto real de los nodos <w:t> (sin duplicación de python-docx)."""
    return "".join((t.text or "") for t in el.iter(W_NS + "t"))


def normalize_text(txt: str) -> str:
    return " ".join(txt.split()).strip()


def replace_sdt_text(el, new_text: str) -> bool:
    """Reemplaza el texto dentro de un <w:sdt> preservando su fuente, tamaño y estilo exactos."""
    t_nodes = list(el.iter(W_NS + "t"))
    if not t_nodes:
        return False
    t_nodes[0].text = new_text
    if new_text.startswith(" ") or new_text.endswith(" "):
        t_nodes[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for extra in t_nodes[1:]:
        extra.text = ""
    return True


def is_safe_empty_placeholder(el) -> bool:
    """Retorna True si el elemento es un <w:sdt> o <w:p> vacío sin saltos, dibujos ni secciones."""
    tag = el.tag.rsplit("}", 1)[-1]
    if tag not in ("sdt", "p"):
        return False
    if normalize_text(get_xml_text(el)):
        return False
    forbidden = (
        W_NS + "br",
        W_NS + "drawing",
        W_NS + "pict",
        W_NS + "sectPr",
        W_NS + "tbl",
        MC_NS + "AlternateContent",
    )
    for ftag in forbidden:
        if el.find(f".//{ftag}") is not None:
            return False
    return True


def set_cell_shading(cell, fill_hex: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    for child in list(tc_pr):
        if child.tag == qn("w:shd"):
            tc_pr.remove(child)
    tc_pr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))


def set_cell_margins(cell, top=80, bottom=80, left=110, right=110) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    for child in list(tc_pr):
        if child.tag == qn("w:tcMar"):
            tc_pr.remove(child)
    tc_mar = OxmlElement("w:tcMar")
    for side, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tc_mar.append(node)
    tc_pr.append(tc_mar)


def set_table_borders(table, border_color: str = "CBD5E1") -> None:
    tbl_pr = table._tbl.tblPr
    for child in list(tbl_pr):
        if child.tag == qn("w:tblBorders"):
            tbl_pr.remove(child)
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="{border_color}"/>'
        f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="0F4C75"/>'
        f'  <w:left w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>'
        f'  <w:insideV w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f"</w:tblBorders>"
    )
    tbl_pr.append(borders)


def set_repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:tblHeader"))


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))


class SectionWriter:
    """Inserta párrafos, listas, recuadros, tablas y figuras ordenadamente tras un encabezado."""

    def __init__(self, doc: Document, anchor_el) -> None:
        self.doc = doc
        self.anchor = anchor_el

    def _place(self, xml_el) -> None:
        self.anchor.addnext(xml_el)
        self.anchor = xml_el

    def _style_run(
        self,
        run,
        *,
        size: float = 10.5,
        bold: bool = False,
        italic: bool = False,
        color: RGBColor = COLOR_BODY,
        highlight: WD_COLOR_INDEX | None = None,
    ) -> None:
        run.font.name = FONT_NAME
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        run.font.color.rgb = color
        if highlight is not None:
            run.font.highlight_color = highlight

    def add_para(
        self,
        parts: str | list[tuple[str, dict]],
        *,
        space_before: float = 2,
        space_after: float = 6,
        align: WD_ALIGN_PARAGRAPH = WD_ALIGN_PARAGRAPH.JUSTIFY,
        left_indent: float = 0.0,
        line_spacing: float = 1.15,
    ) -> None:
        p = self.doc.add_paragraph()
        p.alignment = align
        pf = p.paragraph_format
        pf.space_before = Pt(space_before)
        pf.space_after = Pt(space_after)
        pf.line_spacing = line_spacing
        if left_indent:
            pf.left_indent = Inches(left_indent)

        if isinstance(parts, str):
            r = p.add_run(parts)
            self._style_run(r)
        else:
            for text, kwargs in parts:
                r = p.add_run(text)
                self._style_run(r, **kwargs)
        self._place(p._p)

    def add_subheading(self, title: str) -> None:
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(8)
        pf.space_after = Pt(4)
        pf.keep_with_next = True
        r = p.add_run(title)
        self._style_run(r, size=11.0, bold=True, color=COLOR_PRIMARY)
        self._place(p._p)

    def add_bullet(self, lead_bold: str, rest: str) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.left_indent = Inches(0.28)
        pf.first_line_indent = Inches(-0.15)
        pf.space_before = Pt(1.5)
        pf.space_after = Pt(4)
        pf.line_spacing = 1.13

        rb = p.add_run("• ")
        self._style_run(rb, size=10.5, bold=True, color=COLOR_ACCENT)
        if lead_bold:
            r1 = p.add_run(lead_bold)
            self._style_run(r1, size=10.5, bold=True, color=COLOR_PRIMARY)
        has_todo = "[COMPLETAR" in rest or "[REVISAR" in rest
        r2 = p.add_run(rest)
        self._style_run(
            r2,
            size=10.5,
            bold=has_todo,
            color=RGBColor(0x71, 0x3F, 0x12) if has_todo else COLOR_BODY,
            highlight=WD_COLOR_INDEX.YELLOW if has_todo else None,
        )
        self._place(p._p)

    def add_placeholder(self, text: str) -> None:
        """Inserta un recuadro/párrafo amarillo muy visible para completar datos pendientes."""
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf = p.paragraph_format
        pf.left_indent = Inches(0.15)
        pf.right_indent = Inches(0.15)
        pf.space_before = Pt(4)
        pf.space_after = Pt(6)
        pf.line_spacing = 1.15
        p_pr = p._p.get_or_add_pPr()
        p_pr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="FEF9C3"/>'))
        p_pr.append(
            parse_xml(
                f'<w:pBdr {nsdecls("w")}>'
                f'  <w:left w:val="single" w:sz="18" w:space="6" w:color="CA8A04"/>'
                f'  <w:top w:val="single" w:sz="4" w:space="4" w:color="FDE047"/>'
                f'  <w:bottom w:val="single" w:sz="4" w:space="4" w:color="FDE047"/>'
                f'  <w:right w:val="single" w:sz="4" w:space="4" w:color="FDE047"/>'
                f"</w:pBdr>"
            )
        )
        r = p.add_run(text)
        self._style_run(r, size=10.0, bold=True, color=RGBColor(0x71, 0x3F, 0x12), highlight=WD_COLOR_INDEX.YELLOW)
        self._place(p._p)

    def add_callout(self, title: str, body: str) -> None:
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf = p.paragraph_format
        pf.left_indent = Inches(0.15)
        pf.right_indent = Inches(0.15)
        pf.space_before = Pt(4)
        pf.space_after = Pt(7)
        pf.line_spacing = 1.15
        p_pr = p._p.get_or_add_pPr()
        p_pr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="EFF6FF"/>'))
        p_pr.append(
            parse_xml(
                f'<w:pBdr {nsdecls("w")}>'
                f'  <w:left w:val="single" w:sz="18" w:space="6" w:color="0F4C75"/>'
                f'  <w:top w:val="single" w:sz="4" w:space="4" w:color="BFDBFE"/>'
                f'  <w:bottom w:val="single" w:sz="4" w:space="4" w:color="BFDBFE"/>'
                f'  <w:right w:val="single" w:sz="4" w:space="4" w:color="BFDBFE"/>'
                f"</w:pBdr>"
            )
        )
        r1 = p.add_run(title + " ")
        self._style_run(r1, size=10.0, bold=True, color=COLOR_PRIMARY)
        r2 = p.add_run(body)
        self._style_run(r2, size=10.0, color=COLOR_BODY)
        self._place(p._p)

    def add_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        *,
        col_widths: list[float] | None = None,
        highlight_row_idx: int | None = None,
        highlight_tokens: tuple[str, ...] = ("[COMPLETAR",),
    ) -> None:
        table = self.doc.add_table(rows=1 + len(rows), cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        set_table_borders(table)

        # Encabezado
        hdr_row = table.rows[0]
        set_repeat_header(hdr_row)
        prevent_row_split(hdr_row)
        for ci, htext in enumerate(headers):
            cell = hdr_row.cells[ci]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_shading(cell, "0F4C75")
            set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
            if col_widths:
                cell.width = Inches(col_widths[ci])
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            r = p.add_run(htext)
            self._style_run(r, size=9.5, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))

        # Filas
        for ri, row_vals in enumerate(rows):
            row = table.rows[ri + 1]
            prevent_row_split(row)
            is_hi = highlight_row_idx is not None and ri == highlight_row_idx
            bg_hex = "DBEAFE" if is_hi else ("F8FAFC" if ri % 2 == 1 else "FFFFFF")
            for ci, val in enumerate(row_vals):
                cell = row.cells[ci]
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                has_todo = any(tok in val for tok in highlight_tokens)
                set_cell_shading(cell, "FEF9C3" if has_todo else bg_hex)
                set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
                if col_widths:
                    cell.width = Inches(col_widths[ci])
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.1
                r = p.add_run(val)
                self._style_run(
                    r,
                    size=9.2,
                    bold=(is_hi or ci == 0 or has_todo),
                    color=RGBColor(0x71, 0x3F, 0x12) if has_todo else COLOR_BODY,
                    highlight=WD_COLOR_INDEX.YELLOW if has_todo else None,
                )

        self._place(table._tbl)
        # Pequeño espacio después de la tabla
        sp = self.doc.add_paragraph()
        sp.paragraph_format.space_before = Pt(0)
        sp.paragraph_format.space_after = Pt(4)
        self._place(sp._p)

    def add_figure(self, img_filename: str, caption: str, *, width_in: float = 6.0) -> None:
        img_path = FIG_DIR / img_filename
        if not img_path.exists():
            raise FileNotFoundError(f"No se encontró la figura requerida: {img_path}")

        p_img = self.doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(4)
        p_img.paragraph_format.space_after = Pt(2)
        p_img.paragraph_format.keep_with_next = True
        p_img.add_run().add_picture(str(img_path), width=Inches(width_in))
        self._place(p_img._p)

        p_cap = self.doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(0)
        p_cap.paragraph_format.space_after = Pt(8)
        r = p_cap.add_run(caption)
        self._style_run(r, size=9.0, italic=True, color=COLOR_MUTED)
        self._place(p_cap._p)


def find_section_anchor(body, prefix: str, start_idx: int = 80):
    """Ubica el elemento de encabezado en el cuerpo del documento (después del índice)."""
    target = normalize_text(prefix).lower()
    for el in list(body)[start_idx:]:
        txt = normalize_text(get_xml_text(el)).lower()
        if txt.startswith(target):
            return el
    raise KeyError(f"No se encontró el encabezado '{prefix}' en la plantilla.")


def clean_trailing_empty_placeholder(anchor) -> None:
    """Elimina un único bloque placeholder vacío inmediatamente después del encabezado si es seguro."""
    nxt = anchor.getnext()
    if nxt is not None and is_safe_empty_placeholder(nxt):
        nxt.getparent().remove(nxt)


def update_cover_page(body) -> None:
    """Rellena los controles de portada de Samsung preservando tipografía y posición."""
    replacements = {
        "< título del proyecto >": "NEUMOALERT: Sistema Multimodal de Apoyo al Triage y Detección de Neumonía",
        "< fecha (dd/mm/aa) >": "03/12/26",
        "nombre del equipo": "Equipo 2 — NEUMOALERT (PJ2)",
        "<integrante 1>": "Sebastián Escobar",
        "<integrante 2>": "Jeancarlo Cuesta",
        "<integrante 3>": "Mary Carmen González",
        "<integrante 4>": "Tabita Molina",
        "<integrante 5>": "",
    }
    for el in list(body)[:50]:
        txt = normalize_text(get_xml_text(el)).lower()
        if txt in replacements:
            replace_sdt_text(el, replacements[txt])


def set_cell_text_safely(cell, text: str, *, bold: bool = False, highlight: bool = False) -> None:
    """Escribe en celdas de tablas de la plantilla (que pueden contener <w:sdt> internos)."""
    tc = cell._tc
    sdt_content = tc.find(f".//{W_NS}sdtContent")
    target_p = sdt_content.find(f"{W_NS}p") if sdt_content is not None else tc.find(f"{W_NS}p")

    if target_p is not None:
        # Limpiar runs existentes pero mantener pPr
        for child in list(target_p):
            if child.tag in (W_NS + "r", W_NS + "hyperlink", W_NS + "sdt"):
                target_p.remove(child)
        r = OxmlElement("w:r")
        r_pr = OxmlElement("w:rPr")
        r_fonts = OxmlElement("w:rFonts")
        r_fonts.set(qn("w:ascii"), FONT_NAME)
        r_fonts.set(qn("w:hAnsi"), FONT_NAME)
        r_pr.append(r_fonts)
        sz = OxmlElement("w:sz")
        sz.set(qn("w:val"), "20")  # 10pt
        r_pr.append(sz)
        if bold:
            r_pr.append(OxmlElement("w:b"))
        color = OxmlElement("w:color")
        color.set(qn("w:val"), "713F12" if highlight else "1E293B")
        r_pr.append(color)
        if highlight:
            hi = OxmlElement("w:highlight")
            hi.set(qn("w:val"), "yellow")
            r_pr.append(hi)
        r.append(r_pr)
        t = OxmlElement("w:t")
        t.text = text
        if text.startswith(" ") or text.endswith(" "):
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        r.append(t)
        target_p.append(r)
    else:
        p = cell.paragraphs[0]
        run = p.add_run(text)
        run.font.name = FONT_NAME
        run.font.size = Pt(10)
        run.bold = bold
        if highlight:
            run.font.highlight_color = WD_COLOR_INDEX.YELLOW


def populate_team_review_table(doc: Document) -> None:
    """Ubica la tabla de la Sección 5 por encabezado, rellena los 4 integrantes y quita la 5.ª fila vacía."""
    team_table = None
    for tbl in doc.tables:
        txt = normalize_text(get_xml_text(tbl._tbl)).upper()
        if "REVISIÓN Y COMENTARIOS" in txt and "CATEGORÍA" not in txt:
            team_table = tbl
            break
    if team_table is None:
        raise RuntimeError("No se encontró la tabla de Revisión y Comentarios del Equipo (Sección 5).")

    members = [
        ("Sebastián Escobar", "[COMPLETAR COMENTARIO PERSONAL: reflexión sobre aprendizaje en visión por computador (CNN), desafíos técnicos y trabajo en equipo]"),
        ("Jeancarlo Cuesta", "[COMPLETAR COMENTARIO PERSONAL: reflexión sobre investigación clínica, gestión del proyecto y experiencia en el Capstone]"),
        ("Mary Carmen González", "[COMPLETAR COMENTARIO PERSONAL: reflexión sobre análisis de datos, seguimiento de hitos y aprendizajes del curso]"),
        ("Tabita Molina", "[COMPLETAR COMENTARIO PERSONAL: reflexión sobre diseño clínico multimodal, desarrollo de la aplicación y trabajo colaborativo]"),
    ]
    for idx, (name, prompt) in enumerate(members, start=1):
        row = team_table.rows[idx]
        set_cell_text_safely(row.cells[0], name, bold=True, highlight=False)
        set_cell_text_safely(row.cells[1], prompt, bold=False, highlight=True)

    # Eliminar la 5.ª fila sobrante (índice 5) ya que el Equipo 2 tiene 4 integrantes
    if len(team_table.rows) > 5:
        extra_tr = team_table.rows[5]._tr
        extra_tr.getparent().remove(extra_tr)


def write_all_sections(doc: Document) -> None:
    body = doc.element.body

    def section(prefix: str) -> SectionWriter:
        anchor = find_section_anchor(body, prefix)
        clean_trailing_empty_placeholder(anchor)
        return SectionWriter(doc, anchor)

    # ------------------------------------------------------------------
    # 1.1. Antecedentes
    # ------------------------------------------------------------------
    w = section("1.1. Antecedentes")
    w.add_para(
        "La neumonía adquirida en la comunidad (NAC) constituye uno de los principales problemas de salud pública "
        "a nivel global. Según la Organización Mundial de la Salud (OMS), las infecciones respiratorias bajas "
        "provocan más de 740.000 fallecimientos infantiles cada año y representan una de las causas más frecuentes "
        "de hospitalización y mortalidad en adultos mayores y pacientes con comorbilidades. En Chile, durante los "
        "meses de invierno, los Servicios de Atención Primaria de Urgencia (SAPU), los Servicios de Alta Resolutividad "
        "(SAR) y las unidades de emergencia hospitalaria enfrentan una alta sobrecarga asistencial donde la "
        "identificación oportuna de los casos graves es determinante para el pronóstico del paciente."
    )
    w.add_para(
        "El diagnóstico inicial de neumonía combina la evaluación clínica presencial con la radiografía de tórax (RxT). "
        "Sin embargo, en la práctica cotidiana existen dos cuellos de botella críticos: por una parte, la disponibilidad "
        "de médicos radiólogos en horario inhábil o en centros periféricos es limitada, generando demoras en la entrega "
        "de informes; por otra, las reglas de predicción clínica clásicas —como el reconocido Puntaje de Heckerling "
        "(1990), basado en cinco criterios simples (fiebre > 37,8 °C, frecuencia cardíaca > 100 lpm, crepitaciones, "
        "murmullo vesicular disminuido y ausencia de asma, con AUC original de 0,82)— no incorporan variables de alto "
        "valor predictivo hoy disponibles en cualquier box de urgencias, como la saturación de oxígeno (SatO₂ < 95 %, "
        "Moore et al., 2017), la frecuencia respiratoria, el uso reciente de antipiréticos que enmascaran la fiebre o "
        "síntomas respiratorios altos que sugieren cuadros virales autolimitados."
    )
    w.add_para(
        "Frente a esta brecha, gran parte de las soluciones académicas de inteligencia artificial se enfocan únicamente "
        "en clasificar imágenes radiográficas de forma aislada, desconociendo el contexto clínico del paciente y el "
        "flujo real de trabajo de los servicios de salud. NEUMOALERT surge como una propuesta diferenciadora: un "
        "sistema multimodal de apoyo al triage y a la verificación diagnóstica que articula visión por computador (CNN), "
        "aprendizaje automático sobre ficha clínica validada con asesoría médica (Random Forest) y procesamiento de "
        "lenguaje natural sobre informes radiológicos en español (NLP), bajo un enfoque estrictamente asistencial y "
        "explicable donde la decisión final siempre corresponde al profesional de la salud."
    )

    # ------------------------------------------------------------------
    # 1.2. Motivación y objetivo
    # ------------------------------------------------------------------
    w = section("1.2. Motivación y objetivo")
    w.add_para(
        "La motivación central del equipo es trasladar las técnicas de inteligencia artificial aprendidas durante "
        "Samsung Innovation Campus 2026 hacia una herramienta concreta de alto interés público, diseñada con la "
        "información justa y necesaria que requieren los funcionarios de urgencia (médicos, enfermeras y TENS). "
        "En un entorno clínico real, omitir una neumonía verdadera (falso negativo) implica un riesgo vital mayor que "
        "revisar preventivamente un caso sospechoso (falso positivo); por ello, todo el diseño técnico de NEUMOALERT "
        "prioriza una alta sensibilidad (Recall), transparencia en la explicación de alertas y cero fricción operativa."
    )
    w.add_callout(
        "Objetivo General:",
        "Diseñar, entrenar e integrar un prototipo funcional de nivel TRL 5 (NEUMOALERT) que apoye la priorización "
        "temprana y el doble chequeo diagnóstico de neumonía mediante la fusión multimodal de radiografías de tórax "
        "(CNN), ficha clínica estructurada (Random Forest) e informes radiológicos en español (NLP), desplegado en una "
        "aplicación web responsiva conectada a base de datos relacional.",
    )
    w.add_subheading("Objetivos específicos")
    w.add_bullet(
        "Visión por computador (CNN): ",
        "Entrenar y evaluar un modelo de red neuronal convolucional sobre radiografías reales de tórax (Chest X-Ray "
        "Images — Pneumonia), apuntando a las metas definidas en la Test Card del proyecto (Accuracy > 85 % y "
        "Recall > 94 %).",
    )
    w.add_bullet(
        "Modelo clínico tabular (Random Forest): ",
        "Implementar un clasificador alineado exactamente a la ficha clínica definida con asesoría médica (Puntaje de "
        "Heckerling + predictores de Moore et al. 2017 y signos respiratorios), superando la sensibilidad del puntaje "
        "clásico utilizado de forma aislada.",
    )
    w.add_bullet(
        "NLP clínico en español: ",
        "Desarrollar un pipeline de extracción y clasificación de informes radiológicos en PDF según las cuatro "
        "categorías diagnósticas médicas exactas (Sin neumonía, Posible, Probable y Definitiva), incorporando manejo "
        "de negaciones clínicas y filtro de documentos no pertinentes.",
    )
    w.add_bullet(
        "Fusión multimodal sin circularidad: ",
        "Implementar un motor de decisión adaptable al momento clínico —Modo Triage (60 % CNN + 40 % RF, sin informe "
        "previo) y Modo Doble Chequeo (45 % CNN + 25 % RF + 30 % NLP, con informe disponible)— junto con un detector "
        "automático de discrepancias críticas.",
    )
    w.add_bullet(
        "Aplicación clínica y persistencia: ",
        "Construir una interfaz en Streamlit simplificada para funcionarios de salud, respaldada por base de datos "
        "SQLAlchemy (SQLite/PostgreSQL) y exportación CSV estandarizada para futura validación clínica prospectiva.",
    )

    # ------------------------------------------------------------------
    # 1.3. Integrantes y asignación de roles
    # ------------------------------------------------------------------
    w = section("1.3. Integrantes y asignación de roles")
    w.add_para(
        "El Equipo 2 (PJ2 · Neumonía en radiografías) está conformado por cuatro integrantes con responsabilidades "
        "complementarias que cubren de manera equilibrada las dimensiones de ciencia de datos, ingeniería de software, "
        "investigación clínica y gestión del proyecto:"
    )
    w.add_table(
        headers=["Integrante", "Rol principal", "Responsabilidades y entregables clave"],
        rows=[
            [
                "Sebastián Escobar",
                "Líder de Visión por Computador (CNN)",
                "Preparación y preprocesamiento del dataset público Chest X-Ray; entrenamiento y ajuste de la red "
                "neuronal convolucional (Transfer Learning); cálculo de métricas, matriz de confusión, mapas Grad-CAM "
                "y exportación del modelo para integración.",
            ],
            [
                "Tabita Molina",
                "Líder de Datos Clínicos, NLP, Integración y App Web",
                "Coordinación con asesoría médica (ficha Heckerling + Moore); generación de datasets clínicos "
                "calibrados; entrenamiento de Random Forest y NLP clínico; motor de fusión multimodal; base de datos "
                "SQLAlchemy; aplicación clínica en Streamlit y despliegue en GitHub.",
            ],
            [
                "Jeancarlo Cuesta",
                "Investigación Clínica, Validación y Gestión",
                "Levantamiento epidemiológico y bibliográfico; alineamiento con protocolos de urgencia; diseño y "
                "ejecución de pruebas funcionales; coordinación de hitos y documentación técnica.",
            ],
            [
                "Mary Carmen González",
                "Análisis de Datos, Evaluación y Presentación",
                "Apoyo en análisis exploratorio (EDA) y verificación de métricas contra la rúbrica; seguimiento del "
                "Plan de Acción (WBS); síntesis visual y preparación de la presentación ejecutiva final.",
            ],
        ],
        col_widths=[1.45, 1.65, 3.20],
    )
    w.add_placeholder(
        "[REVISAR CON EL EQUIPO: verificar antes de entregar si desean ajustar algún detalle en la descripción "
        "de roles de Jeancarlo Cuesta y Mary Carmen González.]"
    )

    # ------------------------------------------------------------------
    # 1.4. Cronograma e hitos
    # ------------------------------------------------------------------
    w = section("1.4. Cronograma e hitos")
    w.add_para(
        "El desarrollo del proyecto siguió la planificación oficial del programa Samsung Innovation Campus 2026 "
        "(octubre a diciembre de 2026), estructurada bajo una Estructura de Desglose de Trabajo (WBS) con entregas "
        "incrementales y sesiones de mentoría técnica:"
    )
    w.add_table(
        headers=["Fecha", "Hito / Fase WBS", "Entregable principal", "Estado"],
        rows=[
            ["22 oct 2026", "Conformación y definición del tema", "Equipo 2 (PJ2) formado; selección del tema Neumonía en radiografías.", "Completado"],
            ["29 oct 2026", "Planificación inicial y Test Card", "Borrador de Plan de Acción en Drive; metas objetivo (Accuracy > 85 %, Recall > 94 %).", "Completado"],
            ["17 nov 2026", "Inicio sesiones Capstone y WBS", "Plan de Acción definitivo y WBS validada por el equipo.", "Completado"],
            ["19 nov 2026", "Propuesta técnica y enfoque clínico", "Arquitectura multimodal aprobada e incorporación de la ficha médica real (Heckerling + Moore).", "Completado"],
            ["24 nov 2026", "Modelos base operativos", "Entrenamiento y evaluación de los modelos base: CNN visual, Random Forest clínico y NLP de informes.", "Completado"],
            ["28 nov 2026", "Avance con mentoría e iteraciones", "Corrección de sesgos clínicos (antipiréticos, circularidad, filtro de PDFs) y simplificación UX para salud.", "Completado"],
            ["30 nov 2026", "Demo funcional y repositorio GitHub", "App NEUMOALERT integrada, notebooks en Google Colab y README documentado.", "Completado"],
            ["03 dic 2026", "Entrega y defensa final", "Informe Final oficial SIC 2026 y presentación ejecutiva (7 minutos).", "En cierre"],
        ],
        col_widths=[1.05, 1.70, 2.75, 0.80],
    )

    # ------------------------------------------------------------------
    # 2.1. Obtención de datos
    # ------------------------------------------------------------------
    w = section("2.1. Obtención de datos")
    w.add_para(
        "Dada la naturaleza multimodal de NEUMOALERT, el proyecto articula tres fuentes de datos claramente "
        "diferenciadas según su origen, volumen y propósito de modelado:"
    )
    w.add_table(
        headers=["Fuente / Módulo", "Tipo de datos", "Volumen y composición", "Propósito en NEUMOALERT"],
        rows=[
            [
                "1. Radiografías de tórax (CNN)",
                "Imágenes médicas reales (dataset público Kermany et al., Kaggle)",
                "[COMPLETAR CON DATOS DE SEBASTIÁN: total de imágenes, partición train/val/test y balance Normal vs. Neumonía]",
                "Entrenar y validar el clasificador visual de consolidaciones e infiltrados pulmonares (P_CNN).",
            ],
            [
                "2. Ficha clínica tabular (Random Forest)",
                "Dataset sintético calibrado con evidencia médica (Heckerling 1990; Moore 2017)",
                "2.500 fichas de adultos (≥ 18 años) con 23 variables clínicas y 4 categorías RxT (prevalencia alterada: 28,9 %).",
                "Modelar la probabilidad clínica de radiografía alterada (P_RF) replicando exactamente la ficha del médico asesor.",
            ],
            [
                "3. Informes radiológicos en español (NLP)",
                "Corpus sintético clínico en español + 5 PDFs de prueba multipágina",
                "1.000 informes balanceados (250 por categoría: Sin neumonía, Posible, Probable, Definitiva) con negaciones clínicas.",
                "Clasificar automáticamente conclusiones radiológicas (P_NLP) y detectar discrepancias críticas.",
            ],
        ],
        col_widths=[1.35, 1.45, 1.85, 1.65],
    )
    w.add_callout(
        "Transparencia metodológica y resguardo ético de datos:",
        "Las imágenes utilizadas en la CNN provienen de un repositorio público anonimizado ampliamente validado en la "
        "literatura (Kermany et al., 2018). En cambio, debido a la normativa vigente de confidencialidad de fichas "
        "clínicas (Ley 19.628 / Ley 20.584) y a la imposibilidad de acceder a historias clínicas reales sin aprobación "
        "de un Comité Ético-Científico dentro del plazo académico, los datasets tabular y de informes en español fueron "
        "generados de forma sintética calibrada sobre parámetros epidemiológicos publicados (Heckerling et al., 1990; "
        "Moore et al., 2017; Gennis et al., 1989; Diehr et al., 1984). Su esquema respeta campo a campo la ficha "
        "entregada por el médico asesor, permitiendo reemplazar el origen sintético por fichas reales de urgencia sin "
        "modificar el código del sistema.",
    )

    # ------------------------------------------------------------------
    # 2.2. Metodología de entrenamiento
    # ------------------------------------------------------------------
    w = section("2.2. Metodología de entrenamiento")
    w.add_para(
        "El entrenamiento de los tres modelos se rigió por la metodología CRISP-DM, aplicando separación estricta "
        "entre entrenamiento y prueba estratificada (80 % / 20 %, semilla reproducible random_state = 42), validación "
        "cruzada estratificada de 5 pliegues (5-fold Stratified Cross-Validation) y evaluación final exclusivamente "
        "sobre conjuntos de prueba no vistos (Test Set) para prevenir fugas de información (data leakage)."
    )

    w.add_subheading("A. Modelo de Visión por Computador (CNN para radiografías de tórax)")
    w.add_placeholder(
        "[COMPLETAR CON DATOS DE SEBASTIÁN: describir aquí la arquitectura CNN utilizada (por ejemplo DenseNet121, "
        "ResNet50, EfficientNet o MobileNetV2 con Transfer Learning), estrategia de congelamiento/fine-tuning, "
        "optimizador, learning rate, función de pérdida, manejo del desbalance de clases, épocas y criterio de "
        "selección del umbral de decisión para maximizar Recall.]"
    )

    w.add_subheading("B. Modelo Clínico Tabular (Random Forest sobre Ficha Médica)")
    w.add_para(
        "Siguiendo el diseño metodológico del estudio original de Heckerling (1990), para el entrenamiento y evaluación "
        "binaria de radiografía alterada se excluyeron los 82 casos clasificados como ambiguos ('Posible'), trabajando "
        "sobre 2.418 pacientes evaluables (1.934 en entrenamiento y 484 en prueba no vista, prevalencia positiva de "
        "29,9 %). Se entrenó un ensamble RandomForestClassifier (260 árboles, max_depth = 10, min_samples_leaf = 5) "
        "con ponderación asimétrica class_weight = {0: 1,0, 1: 2,1} para penalizar en mayor medida los falsos "
        "negativos clínicos. Su rendimiento se contrastó sistemáticamente contra dos líneas base: (1) el Puntaje de "
        "Heckerling aislado (umbral ≥ 3) y (2) una Regresión Logística regularizada con estandarización de variables "
        "sobre la misma ficha completa de 23 predictores."
    )

    w.add_subheading("C. Modelo NLP Clínico en Español (TF-IDF + Regresión Logística + Reglas)")
    w.add_para(
        "Sobre los 1.000 informes radiológicos (800 entrenamiento / 200 prueba estratificada), se construyó un "
        "pipeline que combina vectorización TfidfVectorizer con n-gramas de palabras de orden 1 a 3 y escalado "
        "sublineal (sublinear_tf = True) junto a una Regresión Logística multiclase balanceada (class_weight = "
        "'balanced') para clasificar las cuatro categorías médicas exactas (Sin neumonía, Posible, Probable y "
        "Definitiva). El uso de bigramas y trigramas resulta esencial en español clínico porque distingue expresiones "
        "como 'evidencia de neumonía' frente a su negación 'sin evidencia de neumonía'. Sobre la predicción "
        "estadística se incorporó además una capa determinística de seguridad que verifica que el PDF sea "
        "efectivamente un informe radiológico (descartando guías clínicas o papers), detecta antecedentes históricos "
        "de epicrisis previas y extrae prioritariamente la sección de Impresión Diagnóstica."
    )

    w.add_subheading("D. Motor de Fusión Multimodal y Detección de Discrepancias")
    w.add_para(
        "Una decisión crítica de diseño fue evitar la circularidad diagnóstica: en la atención de urgencia real, "
        "el informe del radiólogo no existe todavía durante el triage inicial. Por ello, el motor de fusión opera "
        "automáticamente en dos modos según las fuentes ingresadas:"
    )
    w.add_bullet(
        "Modo Triage (sin informe radiológico previo): ",
        "Combina 60 % de probabilidad visual (CNN) y 40 % de probabilidad clínica (Random Forest). Si solo existe "
        "una de las dos fuentes en ese instante, normaliza los pesos sobre la evidencia disponible sin bloquear al "
        "funcionario.",
    )
    w.add_bullet(
        "Modo Doble Chequeo (cuando ya existe informe PDF): ",
        "Pondera 45 % CNN, 25 % Random Forest y 30 % NLP clínico para contrastar el informe escrito frente a la "
        "imagen y el examen físico.",
    )
    w.add_bullet(
        "Regla de Discrepancia Crítica y Semáforo: ",
        "El sistema clasifica el riesgo en Verde (< 40 %, bajo riesgo), Amarillo (40–69 %, revisión prioritaria) y "
        "Rojo (≥ 70 %, alerta urgente). Adicionalmente, si dos fuentes independientes difieren en más de 40 puntos "
        "porcentuales (por ejemplo, radiografía o clínica muy sospechosa frente a informe normal), se activa una "
        "alerta explícita de DISCREPANCIA CRÍTICA que exige segunda lectura médica.",
    )

    # ------------------------------------------------------------------
    # 2.3. Flujo de trabajo
    # ------------------------------------------------------------------
    w = section("2.3. Flujo de trabajo")
    w.add_para(
        "El flujo de trabajo de NEUMOALERT abarca dos ciclos coordinados: el pipeline técnico de ciencia de datos "
        "(offline, reproducible) y el flujo operativo de atención en urgencias (online, en tiempo real)."
    )
    w.add_bullet(
        "1. Pipeline de datos y modelado: ",
        "Generación y validación de datasets (scripts/generate_datasets.py) → preprocesamiento y validación cruzada "
        "(scripts/train_models.py y notebooks independientes en Google Colab) → serialización de artefactos entrenados "
        "(.joblib para RF y NLP; archivo .keras/.h5/.pt para la CNN) y exportación automática de métricas "
        "(models/model_metrics.json).",
    )
    w.add_bullet(
        "2. Ingreso en box de urgencias: ",
        "El funcionario ingresa los datos básicos del paciente (RUT ficticio en etapa prototipo, edad y sexo), "
        "verifica criterios de inclusión clínica y registra signos vitales junto a los 5 ítems de Heckerling.",
    )
    w.add_bullet(
        "3. Carga de exámenes complementarios: ",
        "De forma opcional y no bloqueante, se adjunta la radiografía de tórax (PNG/JPG) y/o el informe radiológico "
        "en PDF o texto libre.",
    )
    w.add_bullet(
        "4. Inferencia multimodal y explicación: ",
        "En menos de un segundo, NEUMOALERT ejecuta los modelos activos, calcula el semáforo de prioridad, muestra "
        "los factores clínicos más influyentes del caso y alerta si existe discrepancia entre fuentes.",
    )
    w.add_bullet(
        "5. Priorización y trazabilidad clínica: ",
        "El caso se guarda en base de datos relacional y se incorpora inmediatamente a la Cola de Urgencias ordenada "
        "por gravedad (Rojo → Amarillo → Verde), permitiendo luego registrar el informe definitivo del radiólogo y "
        "exportar la cohorte anonimizada en formato CSV.",
    )

    # ------------------------------------------------------------------
    # 2.4. Diseño del sistema
    # ------------------------------------------------------------------
    w = section("2.4. Diseño del sistema")
    w.add_para(
        "El sistema se diseñó bajo una arquitectura modular en Python 3.11 que desacopla completamente la interfaz "
        "de usuario, los motores de inferencia y la capa de almacenamiento, garantizando facilidad de mantenimiento "
        "y escalabilidad institucional:"
    )
    w.add_bullet(
        "Módulos de IA desacoplados (src/): ",
        "clinical_rf.py (ficha médica y Random Forest), nlp_radiology.py (extracción PDF con PyMuPDF, filtro de "
        "documentos y clasificador 4 clases), cnn_adapter.py (conector plug-and-play que detecta automáticamente "
        "modelos .keras, .h5 o .pt en la carpeta models/ sin reescribir la app) y multimodal_fusion.py (reglas de "
        "fusión Triage / Doble Chequeo y alertas de discrepancia).",
    )
    w.add_bullet(
        "Capa de persistencia relacional (src/database.py): ",
        "Implementada con SQLAlchemy sobre un esquema normalizado de cuatro tablas —pacientes, imagenes, "
        "diagnosticos y alertas— que opera con SQLite en entorno local y conmuta automáticamente a PostgreSQL "
        "en la nube mediante la variable de entorno DATABASE_URL.",
    )
    w.add_bullet(
        "Interfaz web responsiva (app.py): ",
        "Desarrollada en Streamlit con diseño limpio de tres pantallas operativas, compatible con despliegue directo "
        "desde GitHub en Streamlit Community Cloud.",
    )
    w.add_figure(
        "fig1_arquitectura_sistema.png",
        "Figura 1. Arquitectura general del sistema multimodal NEUMOALERT: integración de CNN, Random Forest clínico, "
        "NLP de informes PDF, motor de fusión con detector de discrepancias, aplicación Streamlit y persistencia SQLAlchemy.",
        width_in=6.1,
    )

    # ------------------------------------------------------------------
    # 3.1. Preprocesamiento de datos
    # ------------------------------------------------------------------
    w = section("3.1. Preprocesamiento de datos")
    w.add_para(
        "Cada modalidad requirió un pipeline de limpieza y transformación específico para garantizar calidad de "
        "entrada y coherencia clínica:"
    )
    w.add_bullet(
        "Radiografías de tórax (CNN): ",
        "[COMPLETAR CON DATOS DE SEBASTIÁN: indicar redimensionamiento exacto de imágenes, normalización de "
        "intensidad, tratamiento de canales RGB/escala de grises y técnicas de data augmentation aplicadas en "
        "entrenamiento.]",
    )
    w.add_bullet(
        "Datos clínicos tabulares (Random Forest): ",
        "Se verificó ausencia total de valores faltantes en las 2.500 fichas y validez fisiológica de variables "
        "continuas (temperatura axilar, FC, FR, SatO₂ entre 82 % y 100 %, edad ≥ 18 años y días de evolución). "
        "Los signos auscultatorios, síntomas y comorbilidades se codificaron en formato binario (0/1), se calculó "
        "determinísticamente el Puntaje de Heckerling (0 a 5) y se excluyeron los 82 casos 'Posible' (3,3 %) del "
        "entrenamiento binario supervisado para mantener fidelidad metodológica con Heckerling (1990).",
    )
    w.add_bullet(
        "Informes radiológicos en español (NLP): ",
        "La extracción de texto desde PDFs multipágina mediante PyMuPDF se acompaña de normalización de espacios y "
        "saltos de línea preservando tildes del español médico, filtro automático contra textos bibliográficos o guías "
        "clínicas no individuales, segmentación de la sección de Conclusión / Impresión Diagnóstica y vectorización "
        "TF-IDF con n-gramas (1, 3) que conservan la estructura sintáctica de las negaciones.",
    )

    # ------------------------------------------------------------------
    # 3.2. Análisis exploratorio de datos (EDA)
    # ------------------------------------------------------------------
    w = section("3.2. Análisis exploratorio de datos (EDA)")
    w.add_para(
        "El análisis exploratorio permitió verificar el comportamiento epidemiológico de las variables clínicas antes "
        "del modelado. En las 2.500 fichas clínicas, la distribución de las cuatro categorías diagnósticas de la "
        "Sección E corresponde a 1.696 casos 'Sin neumonía' (67,8 %), 541 'Definitiva' (21,6 %), 181 'Probable' "
        "(7,2 %) y 82 'Posible' (3,3 %), resultando en una prevalencia global de radiografía alterada de 28,9 % "
        "(29,9 % sobre los 2.418 casos evaluables)."
    )
    w.add_para(
        "Como se aprecia en la Figura 2, la tasa de radiografía alterada crece de forma estrictamente monótona con "
        "el Puntaje de Heckerling: 4,8 % con puntaje 0 (n = 146), 6,2 % con puntaje 1 (n = 1.006), 23,4 % con "
        "puntaje 2 (n = 629), 53,1 % con puntaje 3 (n = 354), 84,6 % con puntaje 4 (n = 247) y 92,4 % con puntaje 5 "
        "(n = 118). Paralelamente, el histograma de saturación de oxígeno (SatO₂) confirma la fuerte separación "
        "clínica alrededor del umbral de 95 % reportado por Moore et al. (2017): los pacientes sin neumonía se "
        "concentran entre 95 % y 100 %, mientras que los cuadros compatibles con neumonía se desplazan claramente "
        "hacia rangos hipoxémicos inferiores a 95 %."
    )
    w.add_figure(
        "fig2_eda_heckerling.png",
        "Figura 2. Análisis exploratorio del dataset clínico (n = 2.500): (izquierda) tasa observada de radiografía "
        "alterada según Puntaje de Heckerling; (derecha) distribución de SatO₂ (%) y umbral clínico de 95 %.",
        width_in=6.1,
    )
    w.add_placeholder(
        "[COMPLETAR CON DATOS DE SEBASTIÁN: incorporar aquí un breve párrafo de EDA del dataset de imágenes "
        "Chest X-Ray (proporción Normal vs. Neumonía, dimensiones originales de las radiografías y hallazgos "
        "visuales relevantes sobre el desbalance de clases).]"
    )

    # ------------------------------------------------------------------
    # 3.3. Modelado
    # ------------------------------------------------------------------
    w = section("3.3. Modelado")
    w.add_para(
        "A continuación se presentan los resultados de desempeño obtenidos por los tres componentes de inteligencia "
        "artificial de NEUMOALERT evaluados sobre conjuntos de prueba independientes no vistos durante el entrenamiento."
    )

    w.add_subheading("3.3.1. Modelo de Visión por Computador (CNN — Radiografías de Tórax)")
    w.add_table(
        headers=["Métrica en Test Set", "Meta Test Card", "Resultado CNN", "Observación clínica"],
        rows=[
            ["Accuracy (Exactitud)", "> 85,0 %", "[COMPLETAR CON DATOS DE SEBASTIÁN]", "Desempeño global en clasificación binaria"],
            ["Recall (Sensibilidad — Neumonía)", "> 94,0 %", "[COMPLETAR CON DATOS DE SEBASTIÁN]", "Métrica prioritaria para evitar falsos negativos"],
            ["Precision (Valor Predictivo Positivo)", "Referencial", "[COMPLETAR CON DATOS DE SEBASTIÁN]", "Control de falsas alarmas radiológicas"],
            ["F1-Score", "Referencial", "[COMPLETAR CON DATOS DE SEBASTIÁN]", "Equilibrio entre Precision y Recall"],
            ["ROC-AUC", "Referencial", "[COMPLETAR CON DATOS DE SEBASTIÁN]", "Capacidad discriminativa global"],
            ["Falsos Negativos (FN)", "Mínimo posible", "[COMPLETAR CON DATOS DE SEBASTIÁN]", "Casos de neumonía omitidos en Test Set"],
        ],
        col_widths=[1.80, 1.05, 1.80, 1.65],
    )
    w.add_placeholder(
        "[COMPLETAR CON DATOS DE SEBASTIÁN: insertar aquí las gráficas de entrenamiento de la CNN (curvas de "
        "loss/accuracy), su matriz de confusión en Test Set y 1–2 ejemplos de mapas de activación Grad-CAM que "
        "muestren en qué zona pulmonar se enfoca la red.]"
    )

    w.add_subheading("3.3.2. Modelo Clínico Tabular (Random Forest vs. Líneas Base)")
    w.add_para(
        "En el conjunto de prueba no visto (n = 484 pacientes), la comparación entre la regla clínica tradicional "
        "de Heckerling y los modelos entrenados sobre la ficha médica completa demuestra un salto sustantivo en "
        "capacidad de detección:"
    )
    w.add_table(
        headers=["Modelo evaluado (Test Set, n = 484)", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        rows=[
            ["Baseline 1: Puntaje de Heckerling solo (umbral ≥ 3)", "81,8 %", "71,8 %", "64,8 %", "68,1 %", "0,852"],
            ["Baseline 2: Regresión Logística (ficha completa)", "85,7 %", "76,8 %", "75,2 %", "76,0 %", "0,904"],
            ["Modelo elegido: Random Forest (Heckerling + datos extra)", "86,0 %", "76,2 %", "77,2 %", "76,7 %", "0,901"],
        ],
        col_widths=[2.65, 0.72, 0.75, 0.72, 0.72, 0.74],
        highlight_row_idx=2,
    )
    w.add_para(
        "Incorporar las variables complementarias definidas junto al médico asesor elevó el Recall clínico en "
        "+12,4 puntos porcentuales respecto al Puntaje de Heckerling aislado (de 64,8 % a 77,2 %) y el ROC-AUC en "
        "+0,049 puntos (de 0,852 a 0,901; con promedio en validación cruzada 5-fold de ROC-AUC = 0,936 y Recall = "
        "83,5 %). En la matriz de confusión sobre los 484 casos de prueba, el modelo obtuvo 304 verdaderos negativos, "
        "112 verdaderos positivos, 35 falsos positivos y 33 falsos negativos (Figura 3)."
    )
    w.add_figure(
        "fig3_roc_y_matriz_rf.png",
        "Figura 3. Evaluación del modelo clínico Random Forest en Test Set no visto (n = 484): (izquierda) curvas "
        "ROC comparando Heckerling solo vs. Random Forest completo; (derecha) matriz de confusión binaria.",
        width_in=6.1,
    )
    w.add_para(
        "El análisis de importancia de variables (Figura 4) respalda la interpretabilidad médica del modelo: las "
        "señales con mayor peso corresponden exactamente a los predictores con mayor sustento bibliográfico —"
        "Saturación de oxígeno SatO₂ (25,1 %) y Puntaje de Heckerling (18,5 %), seguidos por crepitaciones en "
        "auscultación (10,4 %), temperatura axilar (8,1 %), frecuencia respiratoria (7,8 %), edad (6,5 %), murmullo "
        "disminuido localizado (4,6 %), frecuencia cardíaca (4,3 %) y rinorrea/odinofagia (3,7 %)."
    )
    w.add_figure(
        "fig4_importancia_variables.png",
        "Figura 4. Top 10 variables clínicas más influyentes en las decisiones del modelo Random Forest.",
        width_in=5.8,
    )

    w.add_subheading("3.3.3. Modelo NLP Clínico en Español (Informes Radiológicos — 4 Categorías)")
    w.add_para(
        "Evaluado sobre 200 informes radiológicos independientes (50 por categoría exacta de la Sección E), el "
        "clasificador TF-IDF + Regresión Logística alcanzó una exactitud global de 95,5 %, F1 Macro de 95,5 % y "
        "Recall Macro de 95,5 % (con F1 Macro promedio de 94,1 % en validación cruzada de 5 pliegues):"
    )
    w.add_table(
        headers=["Categoría médica (Sección E)", "Precision", "Recall", "F1-Score", "Soporte (Test)"],
        rows=[
            ["Sin neumonía", "100,0 %", "96,0 %", "98,0 %", "50"],
            ["Posible", "95,9 %", "94,0 %", "95,0 %", "50"],
            ["Probable", "92,2 %", "94,0 %", "93,1 %", "50"],
            ["Definitiva", "94,2 %", "98,0 %", "96,1 %", "50"],
            ["Promedio Macro / Global", "95,6 %", "95,5 %", "95,5 %", "200"],
        ],
        col_widths=[1.90, 1.05, 1.05, 1.05, 1.25],
        highlight_row_idx=4,
    )
    w.add_para(
        "El resultado más relevante desde el punto de vista de seguridad del paciente se observa en la matriz de "
        "confusión 4×4 (Figura 5): se registraron cero inversiones críticas entre 'Sin neumonía' y 'Definitiva' "
        "(ningún informe normal fue clasificado como neumonía definitiva ni viceversa). Los 9 errores totales sobre "
        "200 informes ocurrieron exclusivamente entre clases adyacentes de incertidumbre (por ejemplo, Posible frente "
        "a Probable). Además, los coeficientes del modelo confirman el aprendizaje efectivo de negaciones clínicas "
        "('sin evidencia de', 'ni') frente a marcadores de certeza diagnóstica ('inequívoca', 'alveolar', "
        "'infiltrado neumónico')."
    )
    w.add_figure(
        "fig5_matriz_nlp_4clases.png",
        "Figura 5. Matriz de confusión 4×4 del modelo NLP clínico en español sobre Test Set (n = 200; Accuracy = 95,5 %, "
        "cero inversiones críticas entre Sin neumonía y Definitiva).",
        width_in=4.7,
    )

    # ------------------------------------------------------------------
    # 3.4. Interfaz de usuario
    # ------------------------------------------------------------------
    w = section("3.4. Interfaz de usuario")
    w.add_para(
        "La interfaz de NEUMOALERT fue diseñada en Streamlit priorizando la ergonomía clínica en servicios de "
        "urgencia: pantallas limpias, responsivas (textos completos sin recortes en escritorio y tablet), sin datos "
        "precargados distractores al iniciar y estructurada únicamente en tres vistas operativas:"
    )
    w.add_bullet(
        "1. Ingreso y Evaluación: ",
        "Formulario clínico ágil para el funcionario (datos básicos, criterios de ingreso Sección A, los 5 puntos de "
        "Heckerling con cálculo instantáneo y signos vitales adicionales), carga opcional de radiografía e informe "
        "PDF, tarjeta de semáforo de prioridad (Rojo/Amarillo/Verde), detalle de factores explicativos y registro "
        "en cola con un clic.",
    )
    w.add_bullet(
        "2. Cola de Urgencias: ",
        "Panel de priorización en tiempo real que ordena automáticamente a los pacientes ingresados desde los más "
        "críticos (Rojo y/o con Discrepancia Crítica) hacia los de menor riesgo, permitiendo seguimiento rápido por "
        "el equipo de turno.",
    )
    w.add_bullet(
        "3. Ficha e Informes: ",
        "Vista dedicada al registro de la categoría radiológica oficial (Sección E: Sin neumonía, Posible, Probable "
        "o Definitiva), revisión histórica de pacientes atendidos y descarga directa en CSV compatible con estudios "
        "clínicos.",
    )
    w.add_para(
        "Para demostraciones académicas sin contaminar la operación habitual, la barra lateral incorpora un selector "
        "discreto de 'Modo prueba' que habilita cuatro casos clínicos tipo y cinco informes PDF de prueba. "
        "Adicionalmente, el conector plug-and-play de la CNN permite activar el modelo visual definitivo en segundos "
        "copiando el archivo entrenado en la carpeta models/."
    )
    w.add_placeholder(
        "[COMPLETAR ANTES DE ENTREGAR: agregar aquí el enlace público al repositorio de GitHub y (si aplica) la "
        "URL desplegada en Streamlit Community Cloud, más 1 o 2 capturas de pantalla limpias de la interfaz "
        "NEUMOALERT en funcionamiento.]"
    )

    # ------------------------------------------------------------------
    # 3.5. Pruebas y mejoras
    # ------------------------------------------------------------------
    w = section("3.5. Pruebas y mejoras")
    w.add_para(
        "Durante las iteraciones con asesoría médica, mentorías del curso y pruebas funcionales automatizadas "
        "(validación sintáctica, pruebas de integración con 5 PDFs críticos, Streamlit AppTest y ejecución completa "
        "de dos notebooks reproducibles en Google Colab), se identificaron y resolvieron cinco mejoras sustantivas:"
    )
    w.add_table(
        headers=["Hallazgo detectado en pruebas", "Riesgo / Causa raíz", "Mejora implementada en NEUMOALERT"],
        rows=[
            [
                "Circularidad diagnóstica al usar informes PDF en triage",
                "Al evaluar inicialmente a un paciente en urgencias todavía no existe informe del radiólogo; mezclar "
                "siempre las 3 fuentes no refleja el flujo real.",
                "Separación en Modo Triage (60 % CNN + 40 % RF) y Modo Doble Chequeo (45 % CNN + 25 % RF + 30 % NLP) "
                "con detector automático de Discrepancia Crítica.",
            ],
            [
                "Brecha frente a la ficha médica real de urgencias",
                "Faltaban variables clínicas clave como antipiréticos ≤ 6 h (enmascaran fiebre), rinorrea/odinofagia, "
                "criterios Sección A y las 4 categorías exactas de la Sección E.",
                "Reconstrucción integral de datasets, reentrenamiento de Random Forest y NLP, y exportación CSV "
                "idéntica a la ficha entregada por el médico asesor.",
            ],
            [
                "Falso positivo del NLP al subir guías clínicas PDF (ej. Neumonia.pdf)",
                "Un documento académico menciona decenas de veces la palabra 'neumonía' sin corresponder a un paciente.",
                "Filtro determinístico de validez de documento radiológico + detector de negaciones explícitas y "
                "antecedentes históricos (epicrisis previas).",
            ],
            [
                "Saturación visual y textos recortados en la interfaz inicial",
                "Demasiadas notas explicativas y botones de prueba visibles dificultaban la lectura rápida en pantallas "
                "medianas para personal de salud.",
                "Rediseño completo de app.py: 3 vistas esenciales, formulario vacío por defecto, CSS responsivo con "
                "textos completos y modo prueba oculto en barra lateral.",
            ],
            [
                "Verificación técnica y reproducibilidad externa",
                "Necesidad de validar la ejecución fuera del equipo local y asegurar trazabilidad de métricas.",
                "Creación y ejecución exitosa de dos notebooks autónomos en Google Colab y verificación automatizada "
                "de la app mediante Streamlit AppTest.",
            ],
        ],
        col_widths=[1.75, 2.05, 2.50],
    )

    # ------------------------------------------------------------------
    # 4.1. Logros y beneficios
    # ------------------------------------------------------------------
    w = section("4.1. Logros y beneficios")
    w.add_para(
        "El equipo completó satisfactoriamente un prototipo multimodal integrado de madurez TRL 5 (validado en "
        "entorno simulado realista), logrando articular en un solo sistema el análisis de imágenes, la semiología "
        "clínica validada y el procesamiento de informes médicos en español. Entre sus principales logros y "
        "beneficios proyectados destacan:"
    )
    w.add_bullet(
        "Impacto asistencial en urgencias: ",
        "Facilita la priorización temprana de pacientes con sospecha de neumonía en centros de alta demanda (SAPU, "
        "SAR y hospitales públicos), reduciendo el riesgo de demoras críticas en pacientes hipoxémicos o con "
        "consolidaciones evidentes.",
    )
    w.add_bullet(
        "Ganancia medible en sensibilidad clínica: ",
        "El modelo clínico tabular supera en +12,4 puntos porcentuales el Recall del Puntaje de Heckerling tradicional "
        "(77,2 % vs. 64,8 %) y ofrece explicabilidad inmediata de las variables que explican la alerta.",
    )
    w.add_bullet(
        "Seguridad diagnóstica mediante doble chequeo: ",
        "El clasificador NLP logra 95,5 % de exactitud sobre las 4 categorías médicas con cero inversiones críticas "
        "(Sin neumonía ↔ Definitiva), habilitando un mecanismo automático de alerta ante contradicciones entre "
        "imagen, clínica e informe.",
    )
    w.add_bullet(
        "Sostenibilidad técnica y captura de datos reales: ",
        "La arquitectura modular con SQLAlchemy (SQLite/PostgreSQL), conector plug-and-play para CNN y exportación "
        "CSV alineada al protocolo médico permite recopilar datos estructurados para investigación clínica sin "
        "aumentar la carga administrativa del funcionario.",
    )

    # ------------------------------------------------------------------
    # 4.2. Mejoras futuras (+ Referencias bibliográficas)
    # ------------------------------------------------------------------
    w = section("4.2. Mejoras futuras")
    w.add_para(
        "Para evolucionar desde un prototipo TRL 5 hacia un sistema clínico validado en entorno real (TRL 7–8), se "
        "proyectan las siguientes líneas de continuidad:"
    )
    w.add_bullet(
        "Validación clínica prospectiva y gobernanza ética: ",
        "Someter el protocolo a Comité Ético-Científico para recolectar fichas clínicas e informes radiológicos "
        "reales de urgencias chilenas, reentrenando y calibrando los modelos Random Forest y NLP en cumplimiento "
        "estricto de las Leyes 20.584 y 21.719 de protección de datos personales.",
    )
    w.add_bullet(
        "Interoperabilidad hospitalaria (DICOM / PACS y HL7 FHIR): ",
        "Integrar la recepción automática de radiografías en formato DICOM desde sistemas PACS y el intercambio "
        "estandarizado de signos vitales e informes con la Ficha Clínica Electrónica (FCE).",
    )
    w.add_bullet(
        "Explicabilidad avanzada y calibración por subgrupos: ",
        "Desplegar mapas de calor Grad-CAM en tiempo real junto a valores SHAP por paciente y calibración "
        "probabilística específica para adultos mayores y pacientes inmunocomprometidos.",
    )
    w.add_bullet(
        "Seguridad, control de acceso y monitoreo en producción: ",
        "Incorporar autenticación institucional por roles clínicos, base de datos PostgreSQL gestionada con "
        "auditoría de accesos y monitoreo continuo de deriva de datos (data drift) ante variaciones estacionales "
        "de virus respiratorios.",
    )

    w.add_subheading("Referencias bibliográficas")
    refs = [
        "Diehr P, Wood RW, Bushyhead J, Krueger L, Wolcott B, Tompkins RK. Prediction of pneumonia in outpatients "
        "with acute cough—a statistical approach. Journal of Chronic Diseases. 1984;37(3):215–225.",
        "Gennis P, Gallagher J, Falvo C, Baker S, Than W. Clinical criteria for the detection of pneumonia in adults: "
        "guidelines for ordering chest roentgenograms in the emergency department. Journal of Emergency Medicine. "
        "1989;7(3):263–268.",
        "Heckerling PS, Tape TG, Wigton RS, et al. Clinical prediction rule for pulmonary infiltrates. Annals of "
        "Internal Medicine. 1990;113(9):664–670.",
        "Kermany DS, Goldbaum M, Cai W, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based "
        "Deep Learning (Chest X-Ray Images — Pneumonia). Cell. 2018;172(5):1122–1131.",
        "Moore M, Stuart B, Little P, et al. Predictors of pneumonia in lower respiratory tract infections: 3C "
        "prospective cough complication cohort study. European Respiratory Journal. 2017;50(5):1700434.",
        "Organización Mundial de la Salud (OMS). Neumonía — Notas descriptivas y datos epidemiológicos globales. 2022.",
        "Self WH, Courtney DM, McNaughton CD, Wunderink RG, Kline JA. High discordance of chest x-ray and computed "
        "tomography for detection of pulmonary opacities in ED patients. American Journal of Emergency Medicine. "
        "2013;31(2):401–405.",
    ]
    for ref in refs:
        w.add_para(
            ref,
            space_before=1,
            space_after=3,
            left_indent=0.20,
            line_spacing=1.10,
            align=WD_ALIGN_PARAGRAPH.LEFT,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el Informe Final SIC 2026 IA sobre la plantilla oficial.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE_PATH, help="Ruta a SIC_IA_Capstone_Informe_Final.docx")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Ruta del documento Word generado")
    args = parser.parse_args()

    if not args.template.exists():
        raise FileNotFoundError(f"No se encontró la plantilla oficial en: {args.template}")

    doc = Document(str(args.template))
    update_cover_page(doc.element.body)
    write_all_sections(doc)
    populate_team_review_table(doc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(args.output))
    print("Informe Final generado correctamente en:", args.output)


if __name__ == "__main__":
    main()
