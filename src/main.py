"""NOVA - IR Material Research Tool CLI."""
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.table import Table
from dotenv import load_dotenv

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.graph import NovaAgent
from src.utils.word_export import export_to_word

# Load environment variables
load_dotenv()

console = Console()


NOVA_BANNER = """
[bold magenta]
  +===========================================================+
  |                                                           |
  |     ##    ##   #####   ##    ##    ###                    |
  |     ###   ##  ##   ##  ##    ##   ## ##                   |
  |     ####  ##  ##   ##  ##    ##  ##   ##                  |
  |     ## ## ##  ##   ##  ##    ##  #######                  |
  |     ##  ####  ##   ##   ##  ##   ##   ##                  |
  |     ##   ###   #####     ####    ##   ##                  |
  |                                                           |
  |          [bold white]< News & Investor Analysis System >[/bold white]          |
  |                                                           |
  +===========================================================+
  |                                                           |
  |    [dim]> IR Materials[/dim]          [dim]> AI-Powered Analysis[/dim]       |
  |    [dim]> News Scraping[/dim]         [dim]> Quality Reports[/dim]           |
  |                                                           |
  +===========================================================+
[/bold magenta]
"""

NOVA_SUBTITLE = """
[dim magenta]  -----------------------------------------------------------
           Analyzing Corporate Intelligence from IR Materials
  -----------------------------------------------------------[/dim magenta]
"""


def display_banner():
    """Display the NOVA startup banner."""
    console.print(NOVA_BANNER)
    console.print(NOVA_SUBTITLE)
    console.print()


def create_agent() -> NovaAgent:
    """Create and configure the NOVA agent."""
    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        console.print("[red]Error: GLM_API_KEY not found in environment variables.[/red]")
        console.print("[dim]Please set your API key in .env file[/dim]")
        raise typer.Exit(1)

    # Check for placeholder values
    if api_key in ["your_api_key_here", "your-api-key-here", "YOUR_API_KEY_HERE"]:
        console.print("[red]Error: GLM_API_KEY contains a placeholder value.[/red]")
        console.print("[dim]Please update .env file with your actual Zhipu AI API key[/dim]")
        console.print("[dim]Get your key at: https://open.bigmodel.cn/[/dim]")
        raise typer.Exit(1)

    # OpenAI API key for embeddings (optional but recommended)
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    embedding_provider = "openai" if openai_api_key else "glm"

    if openai_api_key:
        console.print("[dim]Using OpenAI for embeddings[/dim]")
    else:
        console.print("[dim]Using GLM for embeddings (set OPENAI_API_KEY for better results)[/dim]")

    base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
    data_dir = Path(__file__).parent.parent / "data"

    def progress_callback(message: str):
        console.print(f"  [dim]▸[/dim] {message}")

    return NovaAgent(
        api_key=api_key,
        base_url=base_url,
        data_dir=data_dir,
        progress_callback=progress_callback,
        openai_api_key=openai_api_key,
        embedding_provider=embedding_provider,
    )


def main():
    """Main entry point - starts NOVA CLI."""
    display_banner()

    # Create agent
    with console.status("[bold magenta]Initializing NOVA System...[/bold magenta]"):
        try:
            agent = create_agent()
        except Exception as e:
            console.print(f"[red]Failed to initialize: {e}[/red]")
            raise typer.Exit(1)

    console.print("[green]✓[/green] System initialized\n")

    # Get company name
    company = Prompt.ask("[bold magenta]?[/bold magenta] Enter company name to research")

    if not company:
        console.print("[red]No company specified. Exiting.[/red]")
        raise typer.Exit(1)

    # Resolve company
    console.print(f"\n[magenta]Searching for:[/magenta] {company}")

    with console.status("[bold magenta]Resolving company information...[/bold magenta]"):
        result = agent.resolve_company(company)

    if result.get("error"):
        console.print(f"[red]Error: {result['error']}[/red]")
        raise typer.Exit(1)

    company_info = result.get("company_info")
    if not company_info:
        console.print("[red]Could not find company information.[/red]")
        raise typer.Exit(1)

    # Confirm company
    console.print(f"\n[green]✓[/green] Found: [bold]{company_info.official_name}[/bold]")
    console.print(f"[dim]Website: {company_info.website}[/dim]")

    if not Confirm.ask("Is this correct?", default=True):
        console.print("[yellow]Please try a more specific company name.[/yellow]")
        raise typer.Exit(0)

    # Fetch IR materials
    console.print(f"\n[magenta]Fetching IR materials...[/magenta]\n")

    result = agent.fetch_materials(company_info.official_name)

    if result.get("error"):
        error_msg = result['error']
        console.print(f"[red]Error: {error_msg}[/red]")
        
        # Provide helpful suggestions for common errors
        if "No IR materials found" in error_msg:
            console.print("\n[yellow]💡 Suggestions:[/yellow]")
            console.print("  • Check if the company website URL is correct")
            console.print("  • The company may not have a public IR section")
            console.print("  • Try using the full company name (e.g., 'Beta Technologies' instead of 'beta')")
            console.print("  • Check if the website is accessible in your browser")
            console.print("  • DNS errors may indicate the domain doesn't exist")
        elif "DNS" in error_msg or "nodename" in error_msg or "servname" in error_msg:
            console.print("\n[yellow]💡 Suggestions:[/yellow]")
            console.print("  • The domain may not exist or DNS resolution failed")
            console.print("  • Check your internet connection")
            console.print("  • Verify the company website URL is correct")
            console.print("  • Try using the full company name for better matching")
            console.print(f"  • Website found: {company_info.website}")
            if company_info.ir_website:
                console.print(f"  • IR website: {company_info.ir_website}")
        
        raise typer.Exit(1)

    materials = result.get("materials", [])
    chunks = result.get("chunks_indexed", 0)

    # Display summary
    summary_table = Table(title="Materials Summary", show_header=True)
    summary_table.add_column("Material Type", style="magenta")
    summary_table.add_column("Count", style="green")

    type_counts = {}
    for m in materials:
        mtype = m.material_type
        type_counts[mtype] = type_counts.get(mtype, 0) + 1

    for mtype, count in sorted(type_counts.items()):
        summary_table.add_row(mtype, str(count))

    summary_table.add_row("[bold]Total Chunks Indexed[/bold]", f"[bold]{chunks}[/bold]")

    console.print()
    console.print(summary_table)
    console.print()

    # Enter Q&A mode
    console.print(Panel(
        "[bold]Ready for questions![/bold]\n\n"
        "Ask anything about the company's IR materials.\n"
        "Type [bold magenta]quit[/bold magenta] or [bold magenta]exit[/bold magenta] to end the session.",
        title="Q&A Mode",
        border_style="magenta"
    ))

    while True:
        console.print()
        question = Prompt.ask("[bold magenta]?[/bold magenta] Your question")

        if question.lower() in ["quit", "exit", "q"]:
            console.print("\n[dim]Thank you for using NOVA. Goodbye![/dim]\n")
            break

        if not question.strip():
            continue

        # Get answer
        with console.status("[bold magenta]Analyzing materials...[/bold magenta]"):
            result = agent.ask(question)

        if result.get("error"):
            console.print(f"[yellow]Note: {result['error']}[/yellow]")
            continue

        answer = result.get("current_answer", "")
        score = result.get("answer_score", 0)

        if answer:
            # Check quality threshold
            if result.get("meets_quality_threshold"):
                border_style = "green"
                title_color = "green"
            else:
                border_style = "yellow"
                title_color = "yellow"

            console.print()
            console.print(Panel(
                Markdown(answer),
                title=f"[bold {title_color}]Answer[/bold {title_color}] [dim](Quality: {score}/10)[/dim]",
                border_style=border_style,
                padding=(1, 2)
            ))

            if not result.get("meets_quality_threshold"):
                console.print(f"[dim yellow]⚠ Answer quality ({score}/10) below threshold ({agent.quality_threshold}/10)[/dim yellow]")
            
            # Offer Word export
            console.print()
            if Confirm.ask("[bold magenta]Export to Word document?[/bold magenta]", default=False):
                try:
                    company_name = agent.state.get("company_name", "Unknown")
                    output_path = export_to_word(
                        content=answer,
                        company_name=company_name,
                        question=question
                    )
                    console.print(f"[green]✓[/green] Report exported to: [bold]{output_path}[/bold]")
                except ImportError as e:
                    console.print(f"[yellow]⚠ Word export requires python-docx: pip install python-docx[/yellow]")
                except Exception as e:
                    console.print(f"[red]Error exporting to Word: {e}[/red]")
        else:
            console.print("[yellow]No answer generated. Please try a different question.[/yellow]")


if __name__ == "__main__":
    main()
