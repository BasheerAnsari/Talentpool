from pydantic import BaseModel
from typing import Optional


class FileSchema(BaseModel):
    fileName: str
    fileBase64: str


class DocumentsSchema(BaseModel):
    registrationCertificate: Optional[FileSchema] = None
    gstCertificate: Optional[FileSchema] = None


class CompanyInfoSchema(BaseModel):
    companyName: str
    companyId: str
    companyAddress: str
    contact_name: str
    contactNumber: str
    email: Optional[str] = None


class CredentialsSchema(BaseModel):
    username: str
    password: str


class RecruiterSignupSchema(BaseModel):
    companyInfo: CompanyInfoSchema
    credentials: CredentialsSchema
    documents: DocumentsSchema


class RecruiterLoginSchema(BaseModel):
    username: str
    password: str
    