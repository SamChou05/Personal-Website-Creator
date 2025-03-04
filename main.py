from __future__ import annotations as _annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any, List, Dict, Optional

import logfire
from devtools import debug
from httpx import AsyncClient

from pydantic_ai import Agent, ModelRetry, RunContext

from dotenv import load_dotenv

import html.parser
from bs4 import BeautifulSoup

load_dotenv()

# 'if-token-present' means nothing will be sent if you don't have logfire configured
logfire.configure(send_to_logfire="if-token-present")


@dataclass
class Deps:
    client: AsyncClient
    github_token: Optional[str]
    
    @classmethod
    def from_env(cls, client: AsyncClient) -> "Deps":
        """Create Deps instance from environment variables.
        
        Args:
            client: The AsyncClient instance to use.
            
        Returns:
            A Deps instance with values loaded from environment variables.
        """
        github_token = os.getenv("GITHUB_TOKEN")
        return cls(
            client=client,
            github_token=github_token,
        )

github_agent = Agent(
    'anthropic:claude-3-7-sonnet-latest',
    system_prompt=(
        # Repository Analysis Feature
        'You are a helpful assistant that analyzes GitHub repositories and creates a portfolio website for the user. '
        'To analyze repositories, use these tools:\n'
        '- fetch_repo_structure: Get the repository file structure\n'
        '- fetch_file_content: Get content of specific files\n'
        'To avoid context window limits, only analyze 2-5 key files that best represent the project. '
        'Provide a concise summary of the repository and its main components, dependencies, and how the code is organized.\n'

        # Portfolio Generation Feature  
        'You can also create professional portfolio websites for users from Github profiles. '
        'To generate portfolios, use these tools in sequence:\n'
        '- fetch_github_profile: Get the user\'s Github profile data\n'
        '- fetch_repo_structure: Get structure of user\'s repositories\n'
        '- fetch_file_content: Get content from key repository files\n'
        '- generate_portfolio_website: Generate the final portfolio HTML\n'
        '- complete_html_structure: Validate and fix any incomplete HTML structure\n'
        'The portfolio should showcase:\n'
        '- User\'s repositories\n'
        '- Skills (based on repository languages)\n'
        '- Contact information\n'
        '- Professional design with responsive layout\n'

        # Conversation Feature
        'Maintain conversation history to handle follow-up questions and remember context from previous interactions.'
    ),
    deps_type=Deps,
    retries=5,
)

@github_agent.tool
async def fetch_repo_structure(ctx: RunContext[Deps], repo_url: str) -> Dict[str, Any]:
    """Fetch the structure of a GitHub repository.

    Args:
        ctx: The context.
        repo_url: The GitHub repository URL (e.g., https://github.com/username/repo).
    """
    # Extract owner and repo name from URL
    match = re.match(r'https://github.com/([^/]+)/([^/]+)', repo_url)
    if not match:
        raise ModelRetry('Invalid GitHub repository URL format')
    
    owner, repo = match.groups()
    
    headers = {}
    if ctx.deps.github_token:
        headers['Authorization'] = f'token {ctx.deps.github_token}'
    
    with logfire.span('fetching repo structure', repo=f'{owner}/{repo}') as span:
        r = await ctx.deps.client.get(
            f'https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1',
            headers=headers
        )
        
        # Try 'master' branch if 'main' fails
        if r.status_code == 404:
            r = await ctx.deps.client.get(
                f'https://api.github.com/repos/{owner}/{repo}/git/trees/master?recursive=1',
                headers=headers
            )
        
        r.raise_for_status()
        data = r.json()
        span.set_attribute('response', data)
    
    # Filter to only include files (not directories)
    files = [item for item in data.get('tree', []) if item.get('type') == 'blob']
    
    return {
        'owner': owner,
        'repo': repo,
        'files': [{'path': f['path'], 'size': f['size']} for f in files]
    }


@github_agent.tool
async def fetch_file_content(
    ctx: RunContext[Deps], owner: str, repo: str, file_path: str
) -> Dict[str, Any]:
    """Fetch the content of a specific file from a GitHub repository.

    Args:
        ctx: The context.
        owner: The GitHub repository owner.
        repo: The GitHub repository name.
        file_path: The path to the file within the repository.
    """
    headers = {}
    if ctx.deps.github_token:
        headers['Authorization'] = f'token {ctx.deps.github_token}'
    
    with logfire.span('fetching file content', file=f'{owner}/{repo}/{file_path}') as span:
        r = await ctx.deps.client.get(
            f'https://api.github.com/repos/{owner}/{repo}/contents/{file_path}',
            headers=headers
        )
        r.raise_for_status()
        data = r.json()
        span.set_attribute('response_status', r.status_code)
    
    if 'content' not in data:
        raise ModelRetry(f'Could not retrieve content for {file_path}')
    
    import base64
    content = base64.b64decode(data['content']).decode('utf-8')
    
    return {
        'path': file_path,
        'content': content,
        'size': len(content)
    }


@github_agent.tool
async def fetch_github_profile(ctx: RunContext[Deps], username: str) -> Dict[str, Any]:
    """Fetch a GitHub user profile information.

    Args:
        ctx: The context.
        username: The GitHub username.
    """
    headers = {}
    if ctx.deps.github_token:
        headers['Authorization'] = f'token {ctx.deps.github_token}'
    
    with logfire.span('fetching github profile', username=username) as span:
        # Fetch user profile
        r = await ctx.deps.client.get(
            f'https://api.github.com/users/{username}',
            headers=headers
        )
        r.raise_for_status()
        profile_data = r.json()
        
        # Fetch user repositories
        r = await ctx.deps.client.get(
            f'https://api.github.com/users/{username}/repos?sort=updated&per_page=5',
            headers=headers
        )
        r.raise_for_status()
        repos_data = r.json()
        
        span.set_attribute('response_status', r.status_code)
    
    return {
        'profile': {
            'name': profile_data.get('name', username),
            'bio': profile_data.get('bio', ''),
            'avatar_url': profile_data.get('avatar_url', ''),
            'location': profile_data.get('location', ''),
            'blog': profile_data.get('blog', ''),
            'twitter': profile_data.get('twitter_username', ''),
            'followers': profile_data.get('followers', 0),
            'following': profile_data.get('following', 0),
            'public_repos': profile_data.get('public_repos', 0),
        },
        'repos': [
            {
                'name': repo.get('name', ''),
                'description': repo.get('description', ''),
                'language': repo.get('language', ''),
                'stars': repo.get('stargazers_count', 0),
                'forks': repo.get('forks_count', 0),
                'url': repo.get('html_url', ''),
            }
            for repo in repos_data
        ]
    }


@github_agent.tool
async def generate_portfolio_website(
    ctx: RunContext[Deps], profile_data: Dict[str, Any]
) -> Dict[str, str]:
    """Generate a portfolio website for a GitHub user.
    
    Args:
        ctx: The context.
        profile_data: The GitHub profile data, including profile information and repositories.
        
    Returns:
        A dictionary containing the generated HTML.
    """
    # Extract profile information
    profile = profile_data.get('profile', {})
    repos = profile_data.get('repos', [])
    skills = profile_data.get('skills', [])
    
    # Extract basic profile info
    name = profile.get('name', 'GitHub User')
    bio = profile.get('bio', 'Software Developer')
    avatar_url = profile.get('avatar_url', '')
    location = profile.get('location', '')
    blog = profile.get('blog', '')
    twitter = profile.get('twitter_username', '')
    
    # Sort repositories by stars if not already sorted
    if repos:
        try:
            repos = sorted(repos, key=lambda x: x.get('stargazers_count', 0) or x.get('stars', 0), reverse=True)
        except Exception:
            # If sorting fails, keep original order
            pass
    
    # Extract languages from repositories for skills section if not provided
    if not skills:
        languages = {}
        for repo in repos:
            lang = repo.get('language')
            if lang:
                languages[lang] = languages.get(lang, 0) + 1
        
        # Convert to skills format
        skills = [{"name": lang, "level": min(count * 20, 100)} for lang, count in 
                 sorted(languages.items(), key=lambda x: x[1], reverse=True)]
    
    # Define color scheme based on GitHub's colors
    colors = {
        'primary': '#0366d6',
        'secondary': '#6f42c1',
        'dark': '#24292e',
        'light': '#f6f8fa',
        'accent': '#28a745',
        'text': '#24292e',
        'text-light': '#6a737d',
        'border': '#e1e4e8',
    }
    
    # Language colors for skills
    language_colors = {
        'JavaScript': '#f1e05a',
        'TypeScript': '#2b7489',
        'Python': '#3572A5',
        'Java': '#b07219',
        'C#': '#178600',
        'PHP': '#4F5D95',
        'C++': '#f34b7d',
        'Ruby': '#701516',
        'Go': '#00ADD8',
        'Swift': '#ffac45',
        'Kotlin': '#F18E33',
        'Rust': '#dea584',
        'HTML': '#e34c26',
        'CSS': '#563d7c',
        'Shell': '#89e051',
    }
    
    # Create a more sophisticated CSS with animations and responsive design
    css = f"""
        :root {{
            --primary: {colors['primary']};
            --secondary: {colors['secondary']};
            --dark: {colors['dark']};
            --light: {colors['light']};
            --accent: {colors['accent']};
            --text: {colors['text']};
            --text-light: {colors['text-light']};
            --border: {colors['border']};
            --transition: all 0.3s ease;
            --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            --radius: 8px;
        }}
        
        /* Base Styles */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: var(--text);
            background-color: var(--light);
            transition: var(--transition);
        }}
        
        /* Dark Mode */
        body.dark-mode {{
            --light: #0d1117;
            --dark: #c9d1d9;
            --text: #f0f6fc;
            --text-light: #8b949e;
            --border: #30363d;
            color: var(--text);
        }}
        
        .container {{
            width: 100%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        a {{
            color: var(--primary);
            text-decoration: none;
            transition: var(--transition);
        }}
        
        a:hover {{
            color: var(--secondary);
        }}
        
        /* Header Styles */
        header {{
            background-color: var(--primary);
            color: white;
            padding: 60px 0;
            position: relative;
            overflow: hidden;
        }}
        
        header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            opacity: 0.9;
            z-index: 1;
        }}
        
        .header-content {{
            position: relative;
            z-index: 2;
            display: flex;
            align-items: center;
            gap: 40px;
        }}
        
        .profile-image {{
            width: 150px;
            height: 150px;
            border-radius: 50%;
            border: 5px solid white;
            box-shadow: var(--shadow);
            transition: var(--transition);
            object-fit: cover;
        }}
        
        .profile-image:hover {{
            transform: scale(1.05);
        }}
        
        .profile-info {{
            flex: 1;
        }}
        
        .profile-info h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        .profile-bio {{
            font-size: 1.2rem;
            margin-bottom: 20px;
            opacity: 0.9;
        }}
        
        .contact-info {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        
        .contact-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            background-color: rgba(255, 255, 255, 0.2);
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.9rem;
        }}
        
        .contact-item i {{
            font-size: 1rem;
        }}
        
        .contact-item a {{
            color: white;
        }}
        
        .contact-item a:hover {{
            text-decoration: underline;
        }}
        
        .theme-toggle {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: var(--transition);
            z-index: 10;
        }}
        
        .theme-toggle:hover {{
            background: rgba(255, 255, 255, 0.3);
            transform: rotate(15deg);
        }}
        
        /* Section Styles */
        section {{
            padding: 60px 0;
        }}
        
        .section-title {{
            font-size: 2rem;
            margin-bottom: 40px;
            text-align: center;
            position: relative;
        }}
        
        .section-title::after {{
            content: '';
            position: absolute;
            bottom: -10px;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 4px;
            background-color: var(--primary);
            border-radius: 2px;
        }}
        
        /* Skills Section */
        .skills-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
        }}
        
        .skill-item {{
            background-color: white;
            border-radius: var(--radius);
            padding: 20px;
            box-shadow: var(--shadow);
            transition: var(--transition);
            opacity: 0;
            transform: translateY(20px);
        }}
        
        .dark-mode .skill-item {{
            background-color: #1a1f24;
        }}
        
        .skill-item.animated {{
            opacity: 1;
            transform: translateY(0);
        }}
        
        .skill-item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }}
        
        .skill-name {{
            font-weight: bold;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .language-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }}
        
        .skill-bar {{
            height: 10px;
            background-color: var(--border);
            border-radius: 5px;
            overflow: hidden;
        }}
        
        .skill-progress {{
            height: 100%;
            background-color: var(--primary);
            border-radius: 5px;
            transition: width 1s ease-in-out;
        }}
        
        /* Repositories Section */
        .repos-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 30px;
        }}
        
        .repo-card {{
            background-color: white;
            border-radius: var(--radius);
            padding: 25px;
            box-shadow: var(--shadow);
            transition: var(--transition);
            border: 1px solid var(--border);
            height: 100%;
            display: flex;
            flex-direction: column;
            opacity: 0;
            transform: translateY(20px);
        }}
        
        .dark-mode .repo-card {{
            background-color: #1a1f24;
        }}
        
        .repo-card.animated {{
            opacity: 1;
            transform: translateY(0);
        }}
        
        .repo-card:hover {{
            transform: translateY(-10px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .repo-name {{
            font-size: 1.3rem;
            margin-bottom: 10px;
            color: var(--primary);
        }}
        
        .repo-description {{
            margin-bottom: 15px;
            flex-grow: 1;
            color: var(--text-light);
        }}
        
        .repo-meta {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 20px;
            font-size: 0.9rem;
            color: var(--text-light);
        }}
        
        .repo-language {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        
        .repo-link {{
            display: inline-block;
            padding: 8px 16px;
            background-color: var(--primary);
            color: white;
            border-radius: 4px;
            text-align: center;
            transition: var(--transition);
        }}
        
        .repo-link:hover {{
            background-color: var(--secondary);
            color: white;
        }}
        
        /* Footer */
        footer {{
            background-color: var(--dark);
            color: white;
            padding: 30px 0;
            text-align: center;
        }}
        
        .footer-content {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
        }}
        
        .social-links {{
            display: flex;
            gap: 15px;
        }}
        
        .social-link {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            transition: var(--transition);
        }}
        
        .social-link:hover {{
            background-color: var(--primary);
            transform: translateY(-3px);
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .header-content {{
                flex-direction: column;
                text-align: center;
            }}
            
            .contact-info {{
                justify-content: center;
            }}
            
            .section-title {{
                font-size: 1.8rem;
            }}
            
            .repos-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    """
    
    # Generate HTML for skills
    skills_html = ""
    for skill in skills:
        skill_name = skill.get('name', '')
        skill_level = skill.get('level', 50)
        lang_color = language_colors.get(skill_name, '#888')
        
        skills_html += f"""
        <div class="skill-item">
            <div class="skill-name">
                <span class="language-dot" style="background-color: {lang_color}"></span>
                {skill_name}
            </div>
            <div class="skill-bar">
                <div class="skill-progress" style="width: {skill_level}%"></div>
            </div>
        </div>
        """
    
    # Generate HTML for repositories
    repos_html = ""
    for repo in repos[:6]:  # Show up to 6 repositories
        repo_name = repo.get('name', 'Repository')
        description = repo.get('description', '') or repo.get('detailed_description', '')
        language = repo.get('language', '')
        stars = repo.get('stargazers_count', 0) or repo.get('stars', 0)
        forks = repo.get('forks_count', 0) or repo.get('forks', 0)
        url = repo.get('html_url', '') or f"https://github.com/{name}/{repo_name}"
        
        # Get a color for the language
        lang_color = language_colors.get(language, '#888')
        
        # Get a detailed description if available
        detailed_desc = repo.get('detailed_description', '')
        if detailed_desc == description:
            detailed_desc = ''
        
        repos_html += f"""
        <div class="repo-card">
            <h3 class="repo-name">{repo_name}</h3>
            <p class="repo-description">{description}</p>
            {f'<p><small>{detailed_desc}</small></p>' if detailed_desc else ''}
            <div class="repo-meta">
                {f'<div class="repo-language"><span class="language-dot" style="background-color: {lang_color}"></span> {language}</div>' if language else ''}
                <div><i class="fas fa-star"></i> {stars}</div>
                <div><i class="fas fa-code-branch"></i> {forks}</div>
            </div>
            <a href="{url}" class="repo-link" target="_blank">View Repository</a>
        </div>
        """
    
    # Assemble the complete HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="Portfolio website for {name} - GitHub Developer">
    <meta name="keywords" content="portfolio, github, developer, {name}">
    <title>{name} - Portfolio</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>{css}</style>
</head>
<body>
    <button class="theme-toggle" onclick="toggleDarkMode()">
        <i class="fas fa-moon"></i>
    </button>
    
    <header>
        <div class="container">
            <div class="header-content">
                <img src="{avatar_url}" alt="{name}" class="profile-image">
                <div class="profile-info">
                    <h1>{name}</h1>
                    <p class="profile-bio">{bio}</p>
                    <div class="contact-info">
                        {f'<div class="contact-item"><i class="fas fa-map-marker-alt"></i> {location}</div>' if location else ''}
                        {f'<div class="contact-item"><i class="fas fa-globe"></i> <a href="{blog}" target="_blank">{blog}</a></div>' if blog else ''}
                        {f'<div class="contact-item"><i class="fab fa-twitter"></i> <a href="https://twitter.com/{twitter}" target="_blank">@{twitter}</a></div>' if twitter else ''}
                        <div class="contact-item"><i class="fab fa-github"></i> <a href="https://github.com/{name}" target="_blank">GitHub</a></div>
                    </div>
                </div>
            </div>
        </div>
    </header>
    
    <section id="skills">
        <div class="container">
            <h2 class="section-title">Skills</h2>
            <div class="skills-grid">
                {skills_html if skills_html else '<p>No skills data available</p>'}
            </div>
        </div>
    </section>
    
    <section id="projects">
        <div class="container">
            <h2 class="section-title">Featured Projects</h2>
            <div class="repos-grid">
                {repos_html if repos_html else '<p>No repositories available</p>'}
            </div>
        </div>
    </section>
    
    <footer>
        <div class="container">
            <div class="footer-content">
                <p>&copy; {datetime.datetime.now().year} {name} - GitHub Portfolio</p>
                <div class="social-links">
                    <a href="https://github.com/{name}" class="social-link" target="_blank">
                        <i class="fab fa-github"></i>
                    </a>
                    {f'<a href="https://twitter.com/{twitter}" class="social-link" target="_blank"><i class="fab fa-twitter"></i></a>' if twitter else ''}
                    {f'<a href="{blog}" class="social-link" target="_blank"><i class="fas fa-globe"></i></a>' if blog else ''}
                </div>
            </div>
        </div>
    </footer>
    
    <script>
        function toggleDarkMode() {
            document.body.classList.toggle('dark-mode');
            const icon = document.querySelector('.theme-toggle i');
            if (document.body.classList.contains('dark-mode')) {
                icon.className = 'fas fa-sun';
            } else {
                icon.className = 'fas fa-moon';
            }
            localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
        }
        
        // Check for saved dark mode preference
        document.addEventListener('DOMContentLoaded', function() {
            if (localStorage.getItem('darkMode') === 'true') {
                document.body.classList.add('dark-mode');
                document.querySelector('.theme-toggle i').className = 'fas fa-sun';
            }
            
            // Add animation classes with delay
            const elements = document.querySelectorAll('.skill-item, .repo-card');
            elements.forEach((el, index) => {
                setTimeout(() => {
                    el.classList.add('animated');
                }, 100 + (index * 50));
            });
        });
    </script>
</body>
</html>"""
    
    return {
        'html': html
    }


@github_agent.tool
async def complete_html_structure(ctx: RunContext[Deps], html_content: str) -> Dict[str, str]:
    """Check and fix incomplete HTML structure in the provided HTML content.
    
    Args:
        ctx: The context.
        html_content: The HTML content to check and fix.
        
    Returns:
        A dictionary containing the fixed HTML and a message about what was fixed.
    """
    try:
        # Use BeautifulSoup to parse and fix the HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        fixed_html = str(soup)
        fixes_made = []
        
        if not soup.html:
            # Create basic HTML structure if missing
            fixed_html = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<title>Portfolio</title>\n</head>\n<body>\n' + fixed_html + '\n</body>\n</html>'
            fixes_made.append("Added basic HTML structure")
            # Re-parse with the new structure
            soup = BeautifulSoup(fixed_html, 'html.parser')
        else:
            # Check for head and body tags
            if not soup.head:
                html_tag = soup.html
                head_tag = soup.new_tag('head')
                meta_tag = soup.new_tag('meta')
                meta_tag['charset'] = 'UTF-8'
                head_tag.append(meta_tag)
                title_tag = soup.new_tag('title')
                title_tag.string = 'Portfolio'
                head_tag.append(title_tag)
                
                if html_tag.contents:
                    html_tag.insert(0, head_tag)
                else:
                    html_tag.append(head_tag)
                fixes_made.append("Added head tag")
            
            if not soup.body:
                html_tag = soup.html
                body_tag = soup.new_tag('body')
                
                # Move all content after head into body
                for content in list(html_tag.contents):
                    if content != soup.head and content.name != 'head':
                        body_tag.append(content.extract())
                
                html_tag.append(body_tag)
                fixes_made.append("Added body tag")
            
            fixed_html = str(soup)
        
        # Ensure there's at least some CSS
        if not soup.find('style') and not soup.find('link', attrs={'rel': 'stylesheet'}):
            head_tag = soup.head
            style_tag = soup.new_tag('style')
            style_tag.string = """
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            """
            head_tag.append(style_tag)
            fixes_made.append("Added basic CSS")
            fixed_html = str(soup)
        
        # Check for Font Awesome if there are icon classes but no FA link
        if 'fa-' in fixed_html and not soup.find('link', attrs={'href': lambda x: x and 'font-awesome' in x.lower()}):
            head_tag = soup.head
            fa_link = soup.new_tag('link', attrs={
                'rel': 'stylesheet',
                'href': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
            })
            head_tag.append(fa_link)
            fixes_made.append("Added Font Awesome link")
            fixed_html = str(soup)
        
        # Check for unclosed CSS braces in style tags
        for style_tag in soup.find_all('style'):
            if style_tag.string:
                css_content = style_tag.string
                open_braces = css_content.count('{')
                close_braces = css_content.count('}')
                
                if open_braces > close_braces:
                    # Add missing closing braces
                    style_tag.string = css_content + '\n' + ('}' * (open_braces - close_braces))
                    fixes_made.append(f"Added {open_braces - close_braces} missing CSS closing braces")
                    fixed_html = str(soup)
        
        # Check for missing closing tags by comparing the original and fixed HTML
        if len(fixed_html) != len(html_content) and fixed_html != html_content:
            fixes_made.append("Fixed HTML structure and missing tags")
        
        # If no fixes were made but the HTML is valid, just return it
        if not fixes_made:
            fixes_made.append("HTML structure is already valid")
        
        return {
            'html': fixed_html,
            'message': f"HTML structure checked and fixed: {', '.join(fixes_made)}."
        }
    except Exception as e:
        # Log the error but return the original HTML
        print(f"Error in complete_html_structure: {str(e)}")
        return {
            'html': html_content,
            'message': f'Error checking HTML structure: {str(e)}. Original HTML returned.'
        }


async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GitHub Repository Analyzer')
    parser.add_argument('repo_url', help='GitHub repository URL to analyze')
    args = parser.parse_args()
    
    async with AsyncClient() as client:
        deps = Deps.from_env(client)
        
        result = await github_agent.run(
            f"Analyze the repository at {args.repo_url} and provide a summary of its structure and main components.",
            deps=deps
        )
        
        debug(result)
        print('\nRepository Analysis:')
        print(result.data)


if __name__ == '__main__':
    asyncio.run(main())

