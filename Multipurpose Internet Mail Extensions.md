---
aliases:
  - Media Type
  - MIME Type
---
A two-part identifier to label and describe the format of data on the internet. It dictates how content is processed across multiple technologies.

A mimetype is composed of two parts: a ==type== and a ==subtype==, typically separated by a `/`: `type/subtype`. An optional parameter can be added to provide additional details: `type/subtype;parameter=value`
- `text/plain;charset=UTF-8`
- `applicaiton/pdf`
- `image/jpeg`
- `text/html`

There are two classes of type: ==discrete== and ==multipart==. 
- Discrete: Types which represent  single file or medium.
- Multipart: Represents a document comprised of multiple component parts, each of which may have its own individual MIME type... or a multipart type may encapsulate multiple files being sent together in one transaction.

Discrete Types registered with the [[Internet Assigned Numbers Authority|IANA]]
- `application`
- `audio`
- `example`
- `font`
- `image`
- `model`
- `text`
- `video`

Multipart Types
- `message`
- `multipart`

