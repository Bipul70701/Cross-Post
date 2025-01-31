export default async function postToDevTo(frontMatter, body) {
  try {
    // Prepare the request body
    const postData = JSON.stringify({
      article: {
        title: frontMatter.title,
        body_markdown: body,
        published: true,
        tags: frontMatter.title,
        canonical_url: frontMatter.canonical_url,
      },
    });

    // Prepare the request headers
    const headers = new Headers({
      "Content-Type": "application/json",
      "api-key": process.env.DEVTO_API_KEY
    });

    // Prepare the request options
    const requestOptions = {
      method: "POST",
      headers: headers,
      body: postData,
      redirect: "follow",
    };

    
    const response = await fetch("https://dev.to/api/articles", requestOptions);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`API Error: ${errorData.error || response.statusText}`);
    }
    const result = await response.json();
    console.log('Successfully published to Dev.to:', result.url);
    return result;
    //return response;
  } catch (error) {
    process.stderr.write(`Error in postToDevTo: ${error.message}\n`);
  }
}
