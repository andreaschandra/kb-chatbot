"""Define all the schemas used in the chatbot."""

from pydantic import BaseModel, Field


class PointSchema(BaseModel):
    """Use this tool if the user asks about keypoints, main points, main arguments, and summary from the document."""

    keypoints: str = Field(
        description="A concise list of the most important facts, findings, or arguments from the source material."
    )
    summary: str = Field(
        description="A short, readable overview that captures the main message or conclusion."
    )
