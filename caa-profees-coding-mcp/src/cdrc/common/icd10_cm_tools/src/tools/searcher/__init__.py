from functools import cached_property

from pydantic import BaseModel

from cdrc.common.icd10_cm_tools.src.tools.data_config import DEFAULT_ICD10_CM_DATA_PATHS
from cdrc.common.icd10_cm_tools.src.tools.searcher.bm25_search import BM25sICD10Search
from cdrc.common.icd10_cm_tools.src.tools.searcher.corpus import CMMany2OneCorpus
from cdrc.common.icd10_cm_tools.src.tools.searcher.icd10cm import CMCode, CMData, parse_icd10cm

CM_XML_PATH = DEFAULT_ICD10_CM_DATA_PATHS.tabular_file
CM_ORDER_PATH = DEFAULT_ICD10_CM_DATA_PATHS.order_file
CM_SEARCH_INDEX_PATH = DEFAULT_ICD10_CM_DATA_PATHS.search_index_file


class ICD10CMData(BaseModel):
    data: CMData = parse_icd10cm(CM_XML_PATH, CM_ORDER_PATH, None)

    @cached_property
    def codes(self) -> list[CMCode]:
        return list(self.data.list_codes())


cm_data = ICD10CMData()
cm_corpus = CMMany2OneCorpus(codes=cm_data.codes, name="cm_tables")
SEARCHER = BM25sICD10Search(corpus=cm_corpus, index_path=CM_SEARCH_INDEX_PATH)
