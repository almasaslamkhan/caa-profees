import pathlib
import re
import xml.etree.ElementTree as ET
from importlib.resources import files
from typing import Annotated, Any, Iterator, TypeVar

import pandas as pd
from loguru import logger
from pydantic import BaseModel, BeforeValidator, Field, computed_field, field_validator

from cdrc.common.icd10_cm_tools.src.tools.searcher.base import ICD10Code, ICD10IteratorNode

T = TypeVar("T")


def _empty_list_to_none(v: list[T] | None) -> list[T] | None:
    if isinstance(v, list) and len(v) == 0:
        return None
    return v


def _clean_note_text(notes: list[ET.Element]) -> list[str]:
    return [note.text.strip() for note in notes if note.text and note.text.strip()]


def _extract_codes_from_text(text: str) -> list[str]:
    """
    Uses regex to extract ICD-10-CM codes from a given text.
    Codes of type 'T88.0-' will be mapped to 'T88.0X'.

    @TODO handle ranges such P35-P39 for chapters/sections
    """
    text = text.replace(".", "")
    text = text.replace("-,", "X,")
    text = text.replace("-)", "X)")
    codes = re.findall(r"\b[A-Z]\d{2,3}(?:\d+)?X?", text)
    formatted_codes = [f"{code[:3]}.{code[3:]}" if len(code) > 3 else code for code in codes]
    return formatted_codes


class CMModifier(BaseModel):
    notes: list[str] = Field(..., frozen=True)

    @computed_field
    def codes(self) -> list[str]:
        extracted_codes = []
        for note in self.notes:
            extracted_codes.extend(_extract_codes_from_text(note))
        return extracted_codes

    def __len__(self) -> int:
        return len(self.notes)


class CMModifiers(BaseModel, frozen=True):
    includes: Annotated[CMModifier | None, Field(title="Includes", description="These terms are classified here")] = None
    excludes1: Annotated[CMModifier | None, Field(title="Excludes I", description="NEVER use the given code together with these codes")] = None
    excludes2: Annotated[CMModifier | None, Field(title="Excludes II", description="Not included here, may use both codes if applicable")] = None
    code_first: Annotated[CMModifier | None, Field(title="Code first", description="Must code another condition first")] = None
    code_also: Annotated[CMModifier | None, Field(title="Code also", description="Code both conditions")] = None
    use_additional_code: Annotated[CMModifier | None, Field(title="Use additional code", description="Add another code to fully describe")] = None

    @field_validator("*", mode="before")
    @classmethod
    def validate_empty_list(cls, value: CMModifier | list[str] | None) -> CMModifier | None:
        if isinstance(value, list):
            if len(value) == 0:
                return
            else:
                return CMModifier(notes=value)
        if isinstance(value, CMModifier) and len(value.notes) == 0:
            return
        return value


def _is_empty_modifiers(m: CMModifiers | None) -> CMModifiers | None:
    if m is None:
        return
    for modifier in [m.includes, m.excludes1, m.excludes2, m.code_first, m.code_also, m.use_additional_code]:
        if modifier is not None:
            return m
    return


class SevenCharExtension(BaseModel):
    char: str = Field(..., frozen=True, min_length=1, max_length=1)
    text: str = Field(..., frozen=True)


class SevenCharExtensions(BaseModel):
    root_code: str = Field(..., frozen=True)  # root code where it was defined in xml
    note: str = Field(..., frozen=True)
    extensions: list[SevenCharExtension] = Field(..., frozen=None)


class CMCode(ICD10Code, ICD10IteratorNode["CMCode"], frozen=True):
    inclusion_term: Annotated[
        list[str] | None, Field(title="Inclusion terms", description="These specific terms belong to this code"), BeforeValidator(_empty_list_to_none)
    ] = None
    modifiers: Annotated[CMModifiers | None, Field(title="Modifiers"), BeforeValidator(_is_empty_modifiers)] = None
    # visual_impairments: list[VisualImpairment] | None = Field(description="") # exists for just 1 code, seems like table
    seven_char_extensions: SevenCharExtensions | None = Field(default=None)
    diagnoses: Annotated[list["CMCode"] | None, Field(description="Children diagnoses nodes"), BeforeValidator(_empty_list_to_none)] = None
    order_number: int = Field(..., ge=1)
    from_seven_char_notes: bool
    synonyms: Annotated[
        list[str] | None,
        Field(title="Synonyms", description="These terms are synonmous to the main code description"),
        BeforeValidator(_empty_list_to_none),
    ] = None

    def __len__(self) -> int:
        return (sum([len(c) for c in self.diagnoses]) if self.diagnoses is not None else 0) + 1  # self and number of children

    def list_codes(self) -> Iterator["CMCode"]:
        yield self.model_copy(update={"diagnoses": None})
        if self.diagnoses is not None:
            for c in self.diagnoses:
                yield from c.list_codes()


class CMSection(BaseModel, ICD10IteratorNode[CMCode]):
    id: str = Field(..., min_length=3, max_length=7, frozen=True)
    description: str = Field(..., frozen=True)
    notes: Annotated[list[str] | None, Field(default=None, frozen=True), BeforeValidator(_empty_list_to_none)]
    modifiers: Annotated[CMModifiers | None, Field(default=None, frozen=True), BeforeValidator(_is_empty_modifiers)]
    diagnoses: Annotated[list[CMCode] | None, Field(default=None, frozen=True), BeforeValidator(_empty_list_to_none)]  # e.g. C00-C96 has no diagnoses

    def __len__(self) -> int:
        return sum([len(diag) for diag in self.diagnoses]) if self.diagnoses is not None else 0

    def list_codes(self) -> Iterator[CMCode]:
        if self.diagnoses is not None:
            for diag in self.diagnoses:
                yield from diag.list_codes()


class CMChapter(BaseModel, ICD10IteratorNode[CMCode]):
    name: str = Field(..., frozen=True)
    notes: Annotated[list[str] | None, Field(default=None, frozen=True), BeforeValidator(_empty_list_to_none)]
    description: str = Field(..., frozen=True)
    modifiers: Annotated[CMModifiers | None, Field(default=None, frozen=True), BeforeValidator(_is_empty_modifiers)]
    sections: Annotated[list[CMSection] | None, Field(default=None, frozen=True), BeforeValidator(_empty_list_to_none)]

    def __len__(self) -> int:
        return sum([len(sec) for sec in self.sections]) if self.sections else 0

    def list_codes(self) -> Iterator[CMCode]:
        if self.sections is not None:
            for sec in self.sections:
                yield from sec.list_codes()


class CMInstruction(BaseModel):
    title: str = Field(..., frozen=True)
    notes: Annotated[list[str] | None, Field(default=None, frozen=True), BeforeValidator(_empty_list_to_none)]


class CMData(BaseModel, ICD10IteratorNode[CMCode]):
    instructions: list[CMInstruction] = Field(..., frozen=True)
    chapters: list[CMChapter] = Field(..., frozen=True)

    def __len__(self) -> int:
        return sum([len(c) for c in self.chapters])

    def list_codes(self) -> Iterator[CMCode]:
        for chapter in self.chapters:
            yield from chapter.list_codes()


def _parse_modifiers(node: ET.Element) -> CMModifiers:
    return _is_empty_modifiers(
        CMModifiers(
            use_additional_code=_clean_note_text(node.findall("useAdditionalCode/note")),
            excludes1=_clean_note_text(node.findall("excludes1/note")),
            excludes2=_clean_note_text(node.findall("excludes2/note")),
            includes=_clean_note_text(node.findall("includes/note")),
            code_first=_clean_note_text(node.findall("codeFirst/note")),
            code_also=_clean_note_text(node.findall("codeAlso/note")),
        )
    )


def _get_parent_code(code: str) -> str:
    """
    Determines the parent code for a seven-character ICD-10-CM code.
    Handles X placeholders and returns the immediate parent.

    Args:
        code (str): The ICD-10-CM code without decimal point
    Returns:
        str: The parent code
    """
    if len(code) != 7:
        return None

    # Check backwards until we find a non-X character or reach position 3
    for i in range(6, 2, -1):
        if code[i] != "X":
            # Parent is up to this position
            parent_code = code[:i]
            # If parent would end in X, continue looking back
            while parent_code.endswith("X"):
                parent_code = parent_code[:-1]
            return parent_code
    return code[:3]  # Fallback to category level


def _get_formatted_code(code: str) -> str:
    return f"{code[:3]}.{code[3:]}" if len(code) > 3 else code


def parse_order_file(order_file_path: pathlib.Path) -> dict[str, dict]:
    """
    Parses the ICD-10-CM order file and organizes seven-character codes under their immediate parent codes.

    Args:
        order_file_path (str): Path to the order file

    Returns:
        Dict[str, Dict]: Dictionary mapping codes to their data including seven-character children
    """
    code_map = {}
    with files("cdrc.common.icd10_cm_tools.data.source").joinpath(order_file_path).open("r") as f:
        for line in f:
            if len(line) < 15:
                continue

            order_num = line[0:5].strip()
            code = line[6:13].strip()
            valid_flag = line[14:15].strip()

            short_desc = line[16:77].strip()
            assert line[76] == " ", (code, line[76])
            long_desc = line[77:].strip()

            # Add decimal point after third character
            formatted_code = _get_formatted_code(code)

            if len(code) == 7:
                # Get parent code and add this as a seven-character code
                parent_code = _get_parent_code(code)
                formatted_parent = _get_formatted_code(parent_code)

                if formatted_parent in code_map:
                    code_map[formatted_parent]["seven_char_codes"][formatted_code] = {
                        "long_desc": long_desc,
                        "short_desc": short_desc,
                        "valid_code": valid_flag == "1",
                        "order_num": order_num,
                    }
            else:
                # Regular code entry
                code_map[formatted_code] = {
                    "valid_code": valid_flag == "1",
                    "order_num": order_num,
                    "long_desc": long_desc,
                    "short_desc": short_desc,
                    "seven_char_codes": {},
                }

    return code_map


def _parse_seven_char_extensions(node: ET.Element, code: str) -> SevenCharExtensions | None:
    seven_char_note = node.find("sevenChrNote")
    if seven_char_note is not None:
        extensions = []
        for extension in node.findall("sevenChrDef/extension"):
            if extension.text and extension.text.strip():
                extensions.append(SevenCharExtension(char=extension.attrib["char"], text=extension.text.strip()))
        note_element = seven_char_note.find("note")
        note = note_element.text.strip() if note_element is not None and note_element.text else ""
        return SevenCharExtensions(root_code=code, note=note, extensions=extensions)
    return


def _get_new_synonyms(synonyms: list[str], existing_descriptions: list[str]) -> list[str]:
    return list(set(synonyms) - set([desc.lower() for desc in existing_descriptions]))


def _process_synonyms_file(synonyms_path: str) -> dict[str, list[str]]:
    """
    Process the synonyms CSV file into a dictionary mapping codes to their synonyms
    """
    synonyms_df = pd.read_csv(synonyms_path, low_memory=False)
    print("ICD10CM Synonyms were found, adding them to the ontology...")
    # Group all terms by their corresponding codes
    synonyms_map = {}
    for code, terms in synonyms_df.groupby("CODE_VALUE")["TERM"].apply(list).items():
        synonyms_map[code] = list(set(terms))

    return synonyms_map


def _parse_diagnosis(
    node: ET.Element,
    order_data: dict[str, Any],
    synonym_data: dict[str, list[str]] = {},
    parents_modifiers: CMModifiers | None = None,
) -> CMCode:
    code = node.find("name").text.strip()
    desc = node.find("desc").text.strip()
    seven_char_extensions = _parse_seven_char_extensions(node, code)
    synonyms, short_desc = None, None
    modifiers = _parse_modifiers(node)

    diagnoses = node.findall("diag")

    if len(diagnoses) > 0:
        # the children also gets same seven char extensions
        diagnoses = [_parse_diagnosis(child, order_data, synonym_data, parents_modifiers=modifiers or parents_modifiers) for child in diagnoses]

    if node_order_data := order_data.get(code):
        for k, v in node_order_data["seven_char_codes"].items():
            if k not in {d.code for d in diagnoses}:
                if k in synonym_data:
                    synonyms = _get_new_synonyms(synonym_data[k], [v["long_desc"], v["short_desc"]])
                if v["short_desc"].lower() != v["long_desc"].lower():
                    short_desc = v["short_desc"]
                diagnoses.append(
                    CMCode(
                        code=k,
                        description=v["long_desc"],
                        short_description=short_desc,
                        is_valid_code=v["valid_code"],
                        order_number=int(v["order_num"]),
                        from_seven_char_notes=True,
                        modifiers=parents_modifiers,
                        synonyms=synonyms,
                    )
                )
    else:
        parent_code = _get_parent_code(code.replace(".", ""))
        parent_data = order_data[_get_formatted_code(parent_code)]
        node_order_data = parent_data["seven_char_codes"][code]

    # get synonyms
    inclusion_terms = _clean_note_text(node.findall("inclusionTerm/note"))
    if code in synonym_data:
        synonyms = _get_new_synonyms(synonym_data[code], [desc, node_order_data["short_desc"]] + inclusion_terms)
    if node_order_data["short_desc"].lower() != desc.lower():
        short_desc = node_order_data["short_desc"]
    return CMCode(
        code=code,
        description=desc,
        inclusion_term=inclusion_terms,
        modifiers=modifiers,
        diagnoses=diagnoses,
        seven_char_extensions=seven_char_extensions,
        is_valid_code=node_order_data["valid_code"],
        order_number=int(node_order_data["order_num"]),
        short_description=short_desc,
        from_seven_char_notes=False,
        synonyms=synonyms,
    )


def parse_icd10cm(xml_path: pathlib.Path, order_file_path: pathlib.Path, synonyms_path: pathlib.Path | None) -> CMData:
    order_data = parse_order_file(order_file_path)
    with files("cdrc.common.icd10_cm_tools.data.source").joinpath(xml_path).open("r") as f:
        tree = ET.parse(f)
        root = tree.getroot()

    synonym_data = {}
    if synonyms_path:
        synonym_data = _process_synonyms_file(synonyms_path)

    instructions = []
    for instruction in root.findall(".//introSection"):
        instructions.append(
            CMInstruction(
                title=instruction.find("title").text.strip(),
                notes=(
                    [note.text.strip() for note in instruction.findall("instruction/note") if note.text]
                    if instruction.findall("instruction/note")
                    else []
                ),
            )
        )

    chapters = []
    for chapter in root.findall(".//chapter"):
        sections = []
        for section in chapter.findall("section"):
            section_id = section.attrib["id"]
            sections.append(
                CMSection(
                    id=section_id,
                    description=section.find("desc").text.strip(),
                    modifiers=_parse_modifiers(section),
                    diagnoses=[_parse_diagnosis(diag, order_data, synonym_data) for diag in section.findall("diag")],
                    notes=_clean_note_text(section.findall("notes/note")),
                )
            )
        chapters.append(
            CMChapter(
                name=chapter.find("name").text.strip(),
                description=chapter.find("desc").text.strip(),
                modifiers=_parse_modifiers(chapter),
                sections=sections,
                notes=_clean_note_text(chapter.findall("notes/note")),
            )
        )
    data = CMData(instructions=instructions, chapters=chapters)
    logger.debug(f"Loaded ICD-10-CM data with {len(data)} codes.")
    return data
