use chrono::prelude::*;
use regex::Regex;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::fmt::Write as FmtWrite;
use std::process::Command;
use std::time::Duration;

const REPO_PATH: &str = "/opt/el/projetos/workspace/eps-jdk-acesso"; // caminho do repo
const OLLAMA_MODEL: &str = "deepseek-r1:8b"; // modelo pra usar
const OLLAMA_API_URL: &str = "http://localhost:11434/api/generate";
const MAX_DIFF_SIZE: usize = 8000; // maximo do diff
const CHUNK_SIZE: usize = 6000; // tamanho dos pedacos

// Códigos de cores ANSI
const COLOR_RESET: &str = "\x1b[0m";
const COLOR_CYAN: &str = "\x1b[1;36m";
const COLOR_BLUE: &str = "\x1b[1;34m";
const COLOR_YELLOW: &str = "\x1b[1;33m";
const COLOR_GREEN: &str = "\x1b[1;32m";
const COLOR_RED: &str = "\x1b[1;31m";
const COLOR_WHITE: &str = "\x1b[1;37m";
const COLOR_MAGENTA: &str = "\x1b[1;35m";
const COLOR_GRAY: &str = "\x1b[1;90m";

// Etiquetas de log
const LABEL_INFO: &str = "INFO";
const LABEL_REPO: &str = "REPO";
const LABEL_MODELO: &str = "MODELO";
const LABEL_PROCESSANDO: &str = "PROCESSANDO";
const LABEL_SUCESSO: &str = "SUCESSO";
const LABEL_IGNORADO: &str = "IGNORADO";
const LABEL_ERRO: &str = "ERRO";
const LABEL_RESUMO: &str = "RESUMO";
const LABEL_CONCLUIDO: &str = "CONCLUÍDO";
const LABEL_CHUNK: &str = "CHUNK";
const LABEL_PROC: &str = "PROC";
const LABEL_OLLAMA: &str = "OLLAMA";

// Separador
const SEPARATOR: &str = "────────────────────────────────────────────────────────────";

#[derive(Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaResponse {
    response: String,
}

#[derive(Debug, Clone)]
struct CommitInfo {
    hash: String,
    short_hash: String,
    message: String,
    author: String,
    date: String,
    files_changed: Vec<String>,
    insertions: u32,
    deletions: u32,
}

#[derive(Debug)]
struct DiffChunk {
    content: String,
    files: Vec<String>,
    size: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_commits: i32 = env::args()
        .nth(1)
        .unwrap_or_else(|| "10".to_string())
        .parse()?;

    println!("[1;36m[INFO][0m Analisando os últimos {} commits...", num_commits);
    println!("[1;34m[REPO][0m {}", REPO_PATH);
    println!("[1;33m[MODELO][0m {}", OLLAMA_MODEL);
    println!("{}", "-".repeat(60));

    let log_output = Command::new("git")
        .arg("log")
        .arg(format!("-n{}", num_commits))
        .arg("--pretty=format:%H")
        .current_dir(REPO_PATH)
        .output()?;
    
    let hashes: Vec<String> = String::from_utf8(log_output.stdout)?
        .lines()
        .map(String::from)
        .collect();

    let http_client = Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;

    let mut processed = 0;
    let mut skipped = 0;
    let mut errors = 0;

    for (index, hash) in hashes.iter().enumerate() {
        println!("\n{}[{}]{} Commit {}/{}", COLOR_GREEN, LABEL_PROCESSANDO, COLOR_RESET, index + 1, hashes.len());
        
        match process_commit(&http_client, hash) {
            Ok(ProcessResult::Success(filename)) => {
                println!("{}[{}]{} Análise salva em '{}'", COLOR_GREEN, LABEL_SUCESSO, COLOR_RESET, filename);
                processed += 1;
            }
            Ok(ProcessResult::Skipped(reason)) => {
                println!("{}[{}]{} {}", COLOR_YELLOW, LABEL_IGNORADO, COLOR_RESET, reason);
                skipped += 1;
            }
            Err(e) => {
                println!("{}[{}]{} Commit {}: {}", COLOR_RED, LABEL_ERRO, COLOR_RESET, &hash[..12], e);
                errors += 1;
            }
        }
    }

    println!("\n{}", SEPARATOR);
    println!("{}[{}]{}", COLOR_CYAN, LABEL_RESUMO, COLOR_RESET);
    println!("  {}Processados:{} {}", COLOR_GREEN, COLOR_RESET, processed);
    println!("  {}Ignorados:{} {}", COLOR_YELLOW, COLOR_RESET, skipped);
    println!("  {}Erros:{} {}", COLOR_RED, COLOR_RESET, errors);
    println!("{}[{}]{} Análise finalizada!", COLOR_CYAN, LABEL_CONCLUIDO, COLOR_RESET);
    
    Ok(())
}

#[derive(Debug)]
enum ProcessResult {
    Success(String),
    Skipped(String),
}

fn process_commit(client: &Client, hash: &str) -> Result<ProcessResult, Box<dyn std::error::Error>> {
    let commit_info = get_commit_info(hash, REPO_PATH)?;
    let raw_diff = get_commit_diff(hash, REPO_PATH)?;
    
    println!("[1;37mMensagem:[0m {}", commit_info.message);
    println!("[1;35mAutor:[0m {} [1;90mem[0m {}", commit_info.author, commit_info.date);
    println!("[1;32mAlterações:[0m +{} [1;31m-{}[0m linhas em {} arquivo(s)", 
             commit_info.insertions, commit_info.deletions, commit_info.files_changed.len());
    
    // checa se tem mudanca
    if !raw_diff.lines().any(|l| l.starts_with('+') || l.starts_with('-')) {
        return Ok(ProcessResult::Skipped("sem alterações de código detectadas".to_string()));
    }

    let formatted_diff = format_diff_as_markdown(&raw_diff);
    let diff_size = formatted_diff.chars().count();
    
    println!("[1;90mTamanho do diff:[0m {} caracteres", diff_size);

    // processa o diff grande ou normal
    let analysis = if diff_size > MAX_DIFF_SIZE {
        println!("[1;33m[CHUNK][0m Diff muito grande, dividindo em pedaços...");
        process_large_diff(client, &commit_info, &formatted_diff)?
    } else {
        let analysis_prompt = build_analysis_prompt(&commit_info.message, &formatted_diff);
        call_ollama(client, analysis_prompt)?
    };

    let clean_analysis = clean_ollama_response(analysis);

    let final_document = generate_final_document(&commit_info, &clean_analysis, &formatted_diff);

    let filename = generate_filename(&commit_info.message);
    std::fs::write(&filename, final_document)?;
    
    Ok(ProcessResult::Success(filename))
}

fn process_large_diff(client: &Client, commit_info: &CommitInfo, diff: &str) -> Result<String, Box<dyn std::error::Error>> {
    let chunks = split_diff_into_chunks(diff);
    let mut analyses = Vec::new();
    
    println!("[1;33m[CHUNK][0m Dividido em {} pedaços", chunks.len());
    
    for (i, chunk) in chunks.iter().enumerate() {
        println!("[1;34m[PROC][0m Pedaço {}/{}", i + 1, chunks.len());
        
        let chunk_prompt = build_chunk_analysis_prompt(&commit_info.message, &chunk.content, i + 1, chunks.len());
        let chunk_analysis = call_ollama(client, chunk_prompt)?;
        analyses.push(clean_ollama_response(chunk_analysis));
    }
    
    let combined_prompt = build_summary_prompt(&commit_info.message, &analyses);
    let final_analysis = call_ollama(client, combined_prompt)?;
    
    Ok(clean_ollama_response(final_analysis))
}

fn split_diff_into_chunks(diff: &str) -> Vec<DiffChunk> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let mut current_files = Vec::new();
    let mut current_size = 0;
    
    let file_header_re = Regex::new(r"^### Arquivo: `([^`]+)`").unwrap();
    
    for line in diff.lines() {
        let line_size = line.len() + 1;
        
        // se ficou muito grande, cria novo chunk
        if current_size + line_size > CHUNK_SIZE && !current_chunk.is_empty() {
            chunks.push(DiffChunk {
                content: current_chunk.clone(),
                files: current_files.clone(),
                size: current_size,
            });
            
            current_chunk.clear();
            current_files.clear();
            current_size = 0;
        }
        
        // pega nome do arquivo
        if let Some(caps) = file_header_re.captures(line) {
            let file_path = caps.get(1).map_or("", |m| m.as_str());
            if !current_files.contains(&file_path.to_string()) {
                current_files.push(file_path.to_string());
            }
        }
        
        current_chunk.push_str(line);
        current_chunk.push('\n');
        current_size += line_size;
    }
    
    // adiciona ultimo chunk
    if !current_chunk.is_empty() {
        chunks.push(DiffChunk {
            content: current_chunk,
            files: current_files,
            size: current_size,
        });
    }
    
    chunks
}

fn get_commit_info(hash: &str, repo_path: &str) -> Result<CommitInfo, Box<dyn std::error::Error>> {
    let output = Command::new("git")
        .arg("show")
        .arg("-s")
        .arg("--pretty=format:%s%n%an%n%ad%n%b")
        .arg("--date=format:%Y-%m-%d %H:%M")
        .arg(hash)
        .current_dir(repo_path)
        .output()?;

    if !output.status.success() {
        return Err(String::from_utf8(output.stderr)?.into());
    }
    
    let output_str = String::from_utf8(output.stdout)?;
    let lines: Vec<&str> = output_str.lines().collect();
    
    let message = lines.get(0).unwrap_or(&"").to_string();
    let author = lines.get(1).unwrap_or(&"").to_string();
    let date = lines.get(2).unwrap_or(&"").to_string();
    
    let stats_output = Command::new("git")
        .arg("show")
        .arg("--stat")
        .arg("--format=")
        .arg(hash)
        .current_dir(repo_path)
        .output()?;
    
    let stats_str = String::from_utf8(stats_output.stdout)?;
    let (files_changed, insertions, deletions) = parse_git_stats(&stats_str);
    
    Ok(CommitInfo {
        hash: hash.to_string(),
        short_hash: hash[..12].to_string(),
        message,
        author,
        date,
        files_changed,
        insertions,
        deletions,
    })
}

fn parse_git_stats(stats: &str) -> (Vec<String>, u32, u32) {
    let mut files = Vec::new();
    let mut insertions = 0;
    let mut deletions = 0;
    
    for line in stats.lines() {
        if line.contains("|") {
            if let Some(file_part) = line.split('|').next() {
                let file = file_part.trim().to_string();
                if !file.is_empty() {
                    files.push(file);
                }
            }
        } else if line.contains("insertion") || line.contains("deletion") {
            let re = Regex::new(r"(\d+) insertion").unwrap();
            if let Some(caps) = re.captures(line) {
                insertions = caps.get(1).unwrap().as_str().parse().unwrap_or(0);
            }
            
            let re = Regex::new(r"(\d+) deletion").unwrap();
            if let Some(caps) = re.captures(line) {
                deletions = caps.get(1).unwrap().as_str().parse().unwrap_or(0);
            }
        }
    }
    
    (files, insertions, deletions)
}
fn build_analysis_prompt(message: &str, diff: &str) -> String {
    format!(
        "Você é um engenheiro de software sênior especializado em migrações de Java e Hibernate.

CONTEXTO: Migração de Java 8 → 17 e Hibernate 5 → 6

TAREFA: Analise este commit e forneça um resumo estruturado explicando:
1. **PROPÓSITO**: O que esta mudança pretende resolver/implementar
2. **IMPACTO**: Como isso afeta o sistema e a migração
3. **OBSERVAÇÕES**: Pontos importantes, riscos ou considerações

Seja conciso mas informativo. Use linguagem técnica apropriada.

--- MENSAGEM DO COMMIT ---
{}

--- DIFF DO CÓDIGO ---
{}

--- ANÁLISE ---",
        message, diff
    )
}

fn build_chunk_analysis_prompt(message: &str, chunk: &str, chunk_num: usize, total_chunks: usize) -> String {
    format!(
        "Você é um engenheiro de software sênior analisando parte de um commit grande.

CONTEXTO: Migração Java 8→17 e Hibernate 5→6
CHUNK: {}/{} do commit

TAREFA: Analise APENAS este trecho e identifique:
- Principais alterações neste chunk
- Propósito específico das mudanças
- Impacto técnico relevante

Seja conciso. Este é apenas um fragmento de um commit maior.

--- MENSAGEM DO COMMIT ---
{}

--- CHUNK DO DIFF ---
{}

--- ANÁLISE DO CHUNK ---",
        chunk_num, total_chunks, message, chunk
    )
}

fn build_summary_prompt(message: &str, chunk_analyses: &[String]) -> String {
    let combined_analyses = chunk_analyses
        .iter()
        .enumerate()
        .map(|(i, analysis)| format!("**Chunk {}:**\n{}", i + 1, analysis))
        .collect::<Vec<_>>()
        .join("\n\n");

    format!(
        "Você é um engenheiro de software sênior consolidando análises de um commit grande.

CONTEXTO: Migração Java 8→17 e Hibernate 5→6

TAREFA: Com base nas análises dos chunks individuais, crie um resumo consolidado explicando:
1. **PROPÓSITO**: Objetivo geral do commit
2. **IMPACTO**: Efeito conjunto de todas as mudanças
3. **OBSERVAÇÕES**: Pontos importantes da análise completa

--- MENSAGEM DO COMMIT ---
{}

--- ANÁLISES DOS CHUNKS ---
{}

--- RESUMO CONSOLIDADO ---",
        message, combined_analyses
    )
}

fn clean_ollama_response(response: String) -> String {
    let mut cleaned = response;
    
    let think_patterns = [
        "</think>",
        "<think>",
        "</thinking>",
        "<thinking>",
    ];
    
    for pattern in &think_patterns {
        if let Some(pos) = cleaned.find(pattern) {
            if pattern.starts_with("</") {
                let content_start = pos + pattern.len();
                cleaned = cleaned[content_start..].trim_start().to_string();
            } else {
                cleaned = cleaned[..pos].trim_end().to_string();
            }
        }
    }
    
    cleaned.trim().to_string()
}

fn generate_final_document(commit_info: &CommitInfo, analysis: &str, diff: &str) -> String {
    let now = Local::now();
    let formatted_date = now.format("%Y-%m-%d %H:%M:%S").to_string();
    
    format!(
        "# Análise do Commit: {}

## Informações do Commit

**Hash Completo:** `{}`  
**Hash Curto:** `{}`  
**Autor:** {}  
**Data do Commit:** {}  
**Arquivos Modificados:** {}  
**Linhas Adicionadas:** {}  
**Linhas Removidas:** {}  

---

## Análise Técnica

{}

---

## Detalhes das Alterações

{}

---

*Relatório gerado em: {}*",
        commit_info.message.lines().next().unwrap_or("Sem título"),
        commit_info.hash,
        commit_info.short_hash,
        commit_info.author,
        commit_info.date,
        commit_info.files_changed.len(),
        commit_info.insertions,
        commit_info.deletions,
        analysis,
        diff,
        formatted_date
    )
}

fn generate_filename(message: &str) -> String {
    let now = Local::now();
    let date_prefix = now.format("%Y%m%d_%H%M%S").to_string();
    
    let safe_message = message
        .lines()
        .next()
        .unwrap_or("commit")
        .chars()
        .take(40)
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => c,
            ' ' => '_',
            _ => '_',
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string();
    
    if safe_message.is_empty() {
        format!("commit_{}_sem_titulo.md", date_prefix)
    } else {
        format!("commit_{}_{}.md", date_prefix, safe_message)
    }
}

fn call_ollama(client: &Client, prompt: String) -> Result<String, Box<dyn std::error::Error>> {
    let ollama_req = OllamaRequest {
        model: OLLAMA_MODEL,
        prompt,
        stream: false,
    };

    println!("[1;34m[OLLAMA][0m Enviando requisição...");
    
    let res = client
        .post(OLLAMA_API_URL)
        .json(&ollama_req)
        .send()?;

    if !res.status().is_success() {
        return Err(format!("Erro na API do Ollama: {}", res.status()).into());
    }

    let ollama_res: OllamaResponse = res.json()?;
    
    if ollama_res.response.trim().is_empty() {
        return Err("Resposta vazia do Ollama".into());
    }

    println!("[1;32m[OLLAMA][0m Resposta recebida");
    Ok(ollama_res.response)
}

fn format_diff_as_markdown(diff_text: &str) -> String {
    let mut formatted_output = String::new();
    let mut in_diff_block = false;
    let mut current_file = String::new();

    let file_header_re = Regex::new(r"^diff --git a/(.*) b/").unwrap();
    let hunk_header_re = Regex::new(r"^(@@ .* @@)(.*)").unwrap();

    let close_diff_block = |output: &mut String, in_block: &mut bool| {
        if *in_block {
            output.push_str("```\n\n");
            *in_block = false;
        }
    };

    for line in diff_text.lines() {
        if let Some(caps) = file_header_re.captures(line) {
            close_diff_block(&mut formatted_output, &mut in_diff_block);
            
            let file_path = caps.get(1).map_or("", |m| m.as_str());
            if file_path != current_file {
                current_file = file_path.to_string();
                writeln!(formatted_output, "### Arquivo: `{}`", file_path).unwrap();
            }
            
        } else if let Some(caps) = hunk_header_re.captures(line) {
            let _header = caps.get(1).map_or("", |m| m.as_str());
            let _context = caps.get(2).map_or("", |m| m.as_str()).trim();
            
            close_diff_block(&mut formatted_output, &mut in_diff_block);
            
        } else if line.starts_with('+') || line.starts_with('-') || line.starts_with(' ') {
            if !in_diff_block {
                formatted_output.push_str("\n```diff\n");
                in_diff_block = true;
            }
            writeln!(formatted_output, "{}", line).unwrap();
        }
    }

    close_diff_block(&mut formatted_output, &mut in_diff_block);
    formatted_output
}

fn get_commit_message(hash: &str, repo_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("git")
        .arg("show")
        .arg("-s")
        .arg("--pretty=format:%s%n%n%b")
        .arg(hash)
        .current_dir(repo_path)
        .output()?;

    if !output.status.success() {
        return Err(String::from_utf8(output.stderr)?.into());
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn get_commit_diff(hash: &str, repo_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new("git")
        .arg("show")
        .arg(hash)
        .current_dir(repo_path)
        .output()?;

    if !output.status.success() {
        return Err(String::from_utf8(output.stderr)?.into());
    }
    Ok(String::from_utf8(output.stdout)?)
}
