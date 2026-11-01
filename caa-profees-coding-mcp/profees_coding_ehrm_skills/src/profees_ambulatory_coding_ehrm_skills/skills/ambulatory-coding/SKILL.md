---
name: ambulatory-coding
description: "Handles the EHRM ambulatory-coding recode UI event."
metadata:
  mcpServers:
    - name: profees_coding
      endpoint: http://caa-profees-coding-mcp.caa-profees-coding-mcp.svc.cluster.local:80/mcp
      auth: none
      includeTools:
        - recode_profees_coding_ehrm
      observability: true
  eventTriggers:
    - eventType: a2ui.userAction
      actionName: system.updateApplicationContext
      applicationContextGroup: patientchart/documentation
      semanticObject: AmbulatoryCoding
      event: custom.codingAgentRecodeCustomEHRMEvent
      toolCall:
        name: recode_profees_coding_ehrm
        parameterMappings:
          - payloadProperty: patientId
            toolArgument: patient_id
          - payloadProperty: semantic_object_id
            toolArgument: encounter_id
          - payloadProperty: encounterId
            toolArgument: encounter_id
          - payloadProperty: requestId
            toolArgument: request_id
          - payloadProperty: document_ref_id
            toolArgument: document_ref_id
          - payloadProperty: docrefId
            toolArgument: document_ref_id
          - payloadProperty: docref_id
            toolArgument: document_ref_id
---

# Ambulatory Coding

This skill supports only the `custom.codingAgentRecodeCustomEHRMEvent` UI event.
It invokes `recode_profees_coding_ehrm` with the identifiers mapped from that
event payload.
