---
aliases:
  - URI
---
A broad term for something that ==identifies== a resource.

A [[Uniform Resource Locator|URL]] is a type of URI that tells you where the resource is and how to access it.

URI = identifier
URL = locator/access address


Examples:
- URI: s3://user-uploads/users/123/avatar.jpg
	- Identifies an [[Amazon S3|S3]] object
- URL: https://user-uploads.s3.amazonaws.com/users/123/avatar.jpg
	- A URL that tells an HTTP client how to access the object.

All URLs are URIs
Not all URIs are URLs.