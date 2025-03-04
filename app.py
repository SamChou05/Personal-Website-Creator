from __future__ import annotations as _annotations

import asyncio
import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from httpx import AsyncClient

from main import github_agent, Deps
from pydantic_ai import RunContext

# Load environment variables from .env file
load_dotenv()

# Create a directory to store generated portfolios
PORTFOLIO_DIR = Path("generated_portfolios")
PORTFOLIO_DIR.mkdir(exist_ok=True)

# Store conversation history and generated website
class ConversationState:
    def __init__(self):
        self.history = []
        self.client = None
        self.deps = None
        self.generated_html = None
        self.current_username = None
        self.portfolio_path = None
        
    async def initialize(self):
        if self.client is None:
            self.client = AsyncClient()
            self.deps = Deps.from_env(self.client)
    
    async def cleanup(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None
            self.deps = None
    
    def save_portfolio(self):
        """Save the generated portfolio to a file and return the file path."""
        if not self.generated_html or not self.current_username:
            return None
        
        # Create a filename based on the username
        filename = f"{self.current_username}_portfolio.html"
        file_path = PORTFOLIO_DIR / filename
        
        # Save the HTML to a file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.generated_html)
        
        self.portfolio_path = str(file_path)
        return self.portfolio_path


conversation_state = ConversationState()


async def chat_with_agent(message, history):
    # First, immediately add the user message to history and yield to update UI
    history.append([message, "Thinking..."])  # Add user message with "Thinking..." indicator
    yield "", history, None, None  # Update UI immediately to show user message and thinking indicator
    
    try:
        await conversation_state.initialize()
        
        # Gradio's history is a list of [user_message, bot_message] pairs
        # We need to convert this to our internal format for context
        conversation_history = []
        for user_msg, bot_msg in history:
            if user_msg and bot_msg != "Thinking...":  # Skip the thinking indicator in context
                conversation_history.append({"role": "user", "content": user_msg})
            if bot_msg and bot_msg != "Thinking...":
                conversation_history.append({"role": "assistant", "content": bot_msg})
        
        # Create a combined message from history for context
        combined_message = message
        if len(conversation_history) > 1:
            # Include some context from previous exchanges
            context_messages = [
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in conversation_history[-6:-1]  # Last few messages for context
            ]
            if context_messages:
                combined_message = "Previous conversation:\n" + "\n".join(context_messages) + "\n\nCurrent message: " + message
        
        # Check if this is a portfolio generation request
        if "portfolio" in message.lower() and "github.com/" in message.lower():
            # Extract the username from the message if possible
            username_match = re.search(r'github\.com/([a-zA-Z0-9-]+)', message.lower())
            if username_match:
                username = username_match.group(1)
                conversation_state.current_username = username
                history[-1][1] = f"Generating portfolio website for GitHub user: {username}..."
                yield "", history, None, None
                
                try:
                    # Step 1: Fetch the GitHub profile data
                    history[-1][1] = f"Fetching GitHub profile data for {username}..."
                    yield "", history, None, None
                    
                    # Use a retry mechanism for fetching profile data
                    max_retries = 3
                    retry_count = 0
                    profile_data = None
                    
                    while retry_count < max_retries and profile_data is None:
                        try:
                            profile_result = await github_agent.run(
                                f"Fetch the GitHub profile for user {username} using the fetch_github_profile tool and return the complete JSON data.",
                                deps=conversation_state.deps
                            )
                            
                            # Try to extract JSON from the response
                            json_match = re.search(r'```(?:json)?\n(.*?)\n```', profile_result.data, re.DOTALL)
                            if json_match:
                                profile_data_str = json_match.group(1)
                                profile_data = json.loads(profile_data_str)
                            else:
                                # If no JSON block, try to find any dictionary-like structure
                                dict_match = re.search(r'({.*})', profile_result.data, re.DOTALL)
                                if dict_match:
                                    # Clean up the string to make it valid JSON
                                    profile_data_str = dict_match.group(1)
                                    profile_data_str = re.sub(r"'", '"', profile_data_str)
                                    profile_data = json.loads(profile_data_str)
                        except Exception as e:
                            retry_count += 1
                            if retry_count < max_retries:
                                history[-1][1] = f"Retrying profile fetch ({retry_count}/{max_retries})..."
                                yield "", history, None, None
                            else:
                                history[-1][1] = f"Error fetching profile after {max_retries} attempts: {str(e)}. Creating a basic profile instead."
                                yield "", history, None, None
                    
                    # If we couldn't extract profile data, create a basic profile
                    if not profile_data:
                        profile_data = {
                            "profile": {
                                "name": username,
                                "bio": "GitHub User",
                                "avatar_url": f"https://github.com/{username}.png",
                            },
                            "repos": []
                        }
                    
                    # Step 2: Fetch additional repository details for a richer portfolio
                    history[-1][1] = f"Fetching repository details for {username}..."
                    yield "", history, None, None
                    
                    # Get top repositories if available
                    repos = profile_data.get("repos", [])
                    enhanced_repos = []
                    all_languages = {}  # Track all languages for skills section
                    
                    # Sort repositories by stars if available
                    if repos:
                        try:
                            repos = sorted(repos, key=lambda x: x.get("stargazers_count", 0) or x.get("stars", 0), reverse=True)
                        except Exception:
                            # If sorting fails, keep original order
                            pass
                    
                    # Fetch details for up to 5 repositories to keep it manageable
                    for i, repo in enumerate(repos[:5]):
                        if "name" in repo:
                            repo_name = repo["name"]
                            try:
                                # Fetch repository structure to analyze languages and files
                                repo_url = f"https://github.com/{username}/{repo_name}"
                                repo_structure_result = await github_agent.run(
                                    f"Fetch the structure of repository {repo_url} using the fetch_repo_structure tool.",
                                    deps=conversation_state.deps
                                )
                                
                                # Try to extract languages from the repository structure
                                languages_found = False
                                try:
                                    # Look for language information in the response
                                    lang_match = re.search(r'language[s]?.*?:.*?(\w+)', repo_structure_result.data, re.IGNORECASE)
                                    if lang_match:
                                        lang = lang_match.group(1).strip()
                                        if lang and lang.lower() not in ['none', 'unknown']:
                                            # Add to our language collection for skills
                                            all_languages[lang] = all_languages.get(lang, 0) + 1
                                            languages_found = True
                                except Exception:
                                    pass
                                
                                # If we couldn't find languages in the structure, use the repo language
                                if not languages_found and repo.get("language"):
                                    lang = repo.get("language")
                                    all_languages[lang] = all_languages.get(lang, 0) + 1
                                
                                # Get a better description if possible
                                description = repo.get("description", "")
                                if not description:
                                    # Try to extract a description from the README if available
                                    try:
                                        readme_result = await github_agent.run(
                                            f"Fetch the content of the README.md file from repository {repo_url} using the fetch_file_content tool with owner={username}, repo={repo_name}, file_path=README.md",
                                            deps=conversation_state.deps
                                        )
                                        # Extract the first paragraph from the README
                                        readme_match = re.search(r'# .+?\n\n(.+?)\n\n', readme_result.data, re.DOTALL)
                                        if readme_match:
                                            description = readme_match.group(1).strip()
                                            if len(description) > 150:
                                                description = description[:147] + "..."
                                    except Exception:
                                        pass
                                
                                # Create an enhanced repository object with more details
                                enhanced_repo = repo.copy()
                                enhanced_repo["detailed_description"] = description or f"A {repo.get('language', 'code')} project with {repo.get('stargazers_count', 0) or repo.get('stars', 0)} stars."
                                enhanced_repo["languages"] = list(all_languages.keys())
                                enhanced_repos.append(enhanced_repo)
                            except Exception as repo_error:
                                print(f"Error fetching repo details for {repo_name}: {str(repo_error)}")
                                # If fetching details fails, just use the basic repo info
                                enhanced_repos.append(repo)
                    
                    # Extract skills from languages
                    skills = []
                    if all_languages:
                        # Sort languages by frequency
                        skills = [{"name": lang, "level": min(count * 20, 100)} for lang, count in sorted(all_languages.items(), key=lambda x: x[1], reverse=True)]
                    
                    # If we couldn't extract skills, add some default programming skills
                    if not skills and repos:
                        default_languages = set(repo.get("language", "") for repo in repos if repo.get("language"))
                        skills = [{"name": lang, "level": 80} for lang in default_languages if lang]
                    
                    # Update the profile data with enhanced repositories and skills
                    if enhanced_repos:
                        profile_data["repos"] = enhanced_repos
                    profile_data["skills"] = skills
                    
                    # Step 3: Generate a more complex portfolio website
                    history[-1][1] = f"Generating enhanced portfolio website for {username}..."
                    yield "", history, None, None
                    
                    # Create a detailed prompt for generating a complex portfolio
                    portfolio_prompt = f"""
                    Generate a professional, modern portfolio website for GitHub user {username}.
                    
                    Use the generate_portfolio_website tool with this profile data:
                    ```json
                    {json.dumps(profile_data, indent=2)}
                    ```
                    
                    The portfolio should include:
                    1. A modern, responsive design with animations and transitions
                    2. A hero section with the user's profile picture and bio
                    3. A skills section based on the extracted programming languages with skill bars
                    4. A projects section showcasing the top repositories with descriptions, stars, and links
                    5. A contact section with GitHub profile link
                    6. A dark/light mode toggle
                    7. Interactive elements like hover effects on projects
                    8. Custom CSS with a cohesive color scheme
                    9. Font Awesome icons for social links and UI elements
                    
                    Make sure to:
                    - Display the user's top repositories prominently
                    - Show skill levels for each programming language
                    - Include repository stars, forks, and language information
                    - Make the design visually appealing and professional
                    
                    Return ONLY the complete HTML code without any explanations.
                    """
                    
                    # Use a retry mechanism for generating the website
                    max_retries = 3
                    retry_count = 0
                    html_content = None
                    
                    while retry_count < max_retries and html_content is None:
                        try:
                            website_result = await github_agent.run(
                                portfolio_prompt,
                                deps=conversation_state.deps
                            )
                            
                            # Extract HTML from the response
                            html_match = re.search(r'```(?:html)?\n(.*?)\n```', website_result.data, re.DOTALL)
                            if html_match:
                                html_content = html_match.group(1)
                            else:
                                # If no HTML block found, use the entire response
                                # (the agent might have followed instructions to return only HTML)
                                html_content = website_result.data
                                
                                # Verify it looks like HTML
                                if not (html_content.strip().startswith('<!DOCTYPE html>') or 
                                        html_content.strip().startswith('<html') or
                                        '<body' in html_content):
                                    raise ValueError("Response doesn't appear to be valid HTML")
                        except Exception as e:
                            retry_count += 1
                            if retry_count < max_retries:
                                history[-1][1] = f"Retrying website generation ({retry_count}/{max_retries})..."
                                yield "", history, None, None
                            else:
                                history[-1][1] = f"Error generating website after {max_retries} attempts: {str(e)}. Using a simplified template."
                                yield "", history, None, None
                                # Create a basic HTML template as fallback
                                html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{username} - Portfolio</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #0366d6; }}
        .profile {{ display: flex; align-items: center; gap: 20px; margin-bottom: 30px; }}
        .profile img {{ width: 100px; height: 100px; border-radius: 50%; }}
    </style>
</head>
<body>
    <div class="profile">
        <img src="https://github.com/{username}.png" alt="{username}">
        <div>
            <h1>{username}</h1>
            <p>{profile_data.get('profile', {}).get('bio', 'GitHub User')}</p>
        </div>
    </div>
    <h2>GitHub Profile</h2>
    <p><a href="https://github.com/{username}" target="_blank">View on GitHub</a></p>
</body>
</html>"""
                    
                    # Step 4: Validate and fix the HTML structure
                    history[-1][1] = "Validating and fixing HTML structure..."
                    yield "", history, None, None
                    
                    try:
                        # Use the complete_html_structure tool to validate and fix the HTML
                        validation_result = await github_agent.run(
                            f"Check and fix the HTML structure using the complete_html_structure tool with this HTML content:\n```html\n{html_content}\n```",
                            deps=conversation_state.deps
                        )
                        
                        # Try to extract the fixed HTML from the response
                        fixed_html_match = re.search(r'```(?:html)?\n(.*?)\n```', validation_result.data, re.DOTALL)
                        if fixed_html_match:
                            html_content = fixed_html_match.group(1)
                        else:
                            # If no HTML block found, check if the entire response is HTML
                            if validation_result.data.strip().startswith('<!DOCTYPE html>') or validation_result.data.strip().startswith('<html'):
                                html_content = validation_result.data
                    except Exception as e:
                        # Log the error but continue with the original HTML
                        print(f"Error validating HTML: {str(e)}")
                    
                    # Ensure the HTML includes necessary components for a modern website
                    if not re.search(r'<script', html_content, re.IGNORECASE):
                        # Add JavaScript for interactivity if not present
                        html_content = html_content.replace('</body>', """
                        <script>
                            // Dark/Light mode toggle
                            function toggleDarkMode() {
                                document.body.classList.toggle('dark-mode');
                                localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
                            }
                            
                            // Check for saved dark mode preference
                            document.addEventListener('DOMContentLoaded', function() {
                                if (localStorage.getItem('darkMode') === 'true') {
                                    document.body.classList.add('dark-mode');
                                }
                                
                                // Add animation to project cards
                                const cards = document.querySelectorAll('.repo-card');
                                cards.forEach(card => {
                                    card.addEventListener('mouseenter', function() {
                                        this.style.transform = 'translateY(-10px)';
                                    });
                                    card.addEventListener('mouseleave', function() {
                                        this.style.transform = 'translateY(0)';
                                    });
                                });
                            });
                        </script>
                        </body>""")
                    
                    # Add Font Awesome if not present
                    if not re.search(r'font-awesome|fontawesome', html_content, re.IGNORECASE):
                        html_content = html_content.replace('</head>', """
                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                        </head>""")
                    
                    # Add meta tags for better SEO if not present
                    if not re.search(r'<meta name="description"', html_content, re.IGNORECASE):
                        html_content = html_content.replace('<head>', f"""<head>
                        <meta name="description" content="Portfolio website for GitHub user {username}">
                        <meta name="keywords" content="portfolio, github, developer, {username}">""")
                    
                    conversation_state.generated_html = html_content
                    
                    # Save the portfolio to a file
                    portfolio_path = conversation_state.save_portfolio()
                    
                    # Update the message
                    history[-1][1] = f"Enhanced portfolio website for {username} has been generated! You can view it below and download the HTML file."
                except Exception as e:
                    history[-1][1] = f"Error generating portfolio: {str(e)}"
                    yield "", history, None, None
                
                # Final yield with HTML content
                yield "", history, conversation_state.generated_html or None, conversation_state.portfolio_path
                return
        
        # For non-portfolio requests, run the agent normally
        result = await github_agent.run(combined_message, deps=conversation_state.deps)
        
        # Update the last history item with the bot's response
        history[-1][1] = result.data
        
    except Exception as e:
        # Handle any errors and show them in the chat
        error_message = f"Error: {str(e)}"
        history[-1][1] = error_message
    
    # Final yield for non-portfolio requests
    yield "", history, None, None


def download_portfolio():
    """Return the path to the generated portfolio file for download."""
    if conversation_state.portfolio_path and os.path.exists(conversation_state.portfolio_path):
        return conversation_state.portfolio_path
    return None


async def on_close():
    await conversation_state.cleanup()


# Create the Gradio interface
with gr.Blocks(title="GitHub Repository Analyzer") as demo:
    gr.Markdown("# GitHub Repository Analyzer")
    gr.Markdown("""
    This tool helps you analyze GitHub repositories and generate portfolio websites. You can:
    - Paste a GitHub repository URL to analyze its structure
    - Ask questions about the code organization
    - Request to see specific files
    - Generate a portfolio website from a GitHub profile
    
    Examples:
    - "Analyze the repository at https://github.com/username/repo"
    - "Create a portfolio website for https://github.com/username"
    """)
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Enter your message here...", label="Your message")
    clear = gr.Button("Clear conversation")
    
    # HTML output for the generated website
    with gr.Accordion("Generated Portfolio Website", open=False) as portfolio_section:
        html_output = gr.HTML(label="Portfolio Website")
        download_button = gr.Button("Download Portfolio HTML")
        file_output = gr.File(label="Portfolio HTML File", visible=False)
    
    # Use the streaming version of the submit function with the HTML output
    msg.submit(
        fn=chat_with_agent,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, html_output, file_output],
        api_name="chat"
    )
    
    # Download button functionality
    download_button.click(
        fn=download_portfolio,
        inputs=[],
        outputs=[file_output]
    )
    
    # Clear button should also clear the HTML output and file output
    def clear_all():
        conversation_state.generated_html = None
        conversation_state.portfolio_path = None
        conversation_state.current_username = None
        return [], None, None
    
    clear.click(clear_all, None, [chatbot, html_output, file_output], queue=False)
    
    demo.load(lambda: None)
    demo.close(on_close)


# Launch the app
if __name__ == "__main__":
    demo.launch() 