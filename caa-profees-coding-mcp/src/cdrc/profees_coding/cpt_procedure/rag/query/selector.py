from abc import ABC, abstractmethod
from typing import Any, Dict, Union


class BaseQueryPartSelector(ABC):
    @abstractmethod
    def select(self, query_parts: Dict[str, str], **kwargs) -> Union[str, Dict[str, str]]:
        """
        Args:
            query_parts: a dictionary of each part of query and their content
        Returns:
            a query string constructed by the selected query parts
        """
        pass


class QueryPartSelector(BaseQueryPartSelector):
    def __init__(self, config: Dict[str, Any]):
        self.query_parts_lookups = {}
        for stage in config.keys():
            self.query_parts_lookups[stage] = config.get(stage, {})

    def select(self, query_parts: Union[str, Dict[str, str]], stage: str) -> str:
        if not isinstance(query_parts, dict):
            return query_parts
        if stage not in self.query_parts_lookups:
            raise ValueError(f"Unknown query stage: {stage}")

        rule = self.query_parts_lookups[stage]
        if rule is None:
            raise ValueError(f"Query part selection rules for {stage} must not be None.")

        query_mode = rule.get("mode", "all")
        if query_mode == "all":
            selected = str(query_parts)
        elif query_mode == "assessment_and_plan_only":
            selected = str(f"Assessment and Plan: {query_parts.get('assessment_and_plan', {})}")
        else:
            raise ValueError(f"Unknown query mode: {query_mode}")
        return selected
