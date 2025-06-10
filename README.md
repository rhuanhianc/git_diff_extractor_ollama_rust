# Git Diff Extractor com Análise IA

Ferramenta em Rust que extrai diffs de commits do Git e usa o Ollama pra gerar análises técnicas automaticas.

## Autor
**Rhuan Hianc** - Desenvolvedor

## Como usar

### Pré-requisitos
- Rust instalado
- Ollama rodando na porta 11434
- Modelo DeepSeek R1: `ollama pull deepseek-r1:8b`

### Configuração
Edite o caminho do repo no `src/main.rs`:
```rust
const REPO_PATH: &str = "/seu/caminho/aqui";
```

### Executar
```bash
# Últimos 10 commits
cargo run

# Últimos 5 commits  
cargo run 5
```

## Funcionalidades

- Análise automatica de commits
- Divisão de diffs grandes em pedaços menores
- Saída colorida no terminal
- Geração de relatórios em markdown
- Tratamento de erros robusto

## Configurações

- `MAX_DIFF_SIZE`: 8000 caracteres
- `CHUNK_SIZE`: 6000 caracteres  
- `TIMEOUT`: 5 minutos

## Saída

Gera arquivos `.md` com:
- Informações do commit
- Análise técnica (gerada pelo Ollama)
- Diff formatado

### Exemplo de arquivo gerado:
```
commit_20241210_143022_correcao_imports.md
```

## Commits grandes

Para diffs > 8000 caracteres:
1. Divide em chunks automaticamente
2. Analisa cada pedaço separadamente  
3. Consolida as análises no final

## Estatísticas

```
[RESUMO]
  Processados: 8
  Ignorados: 1  
  Erros: 1
```

---

*Otimizado para migrações Java 8→17 e Hibernate 5→6*