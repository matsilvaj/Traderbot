import os
import json
import yaml
import subprocess
import asyncio
from datetime import datetime, time, date
from pathlib import Path

# Imports do Telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Import para ler o .env
from dotenv import load_dotenv

# =====================================================================
# 1. DEFINIÇÃO DE CAMINHOS
# =====================================================================
# Pega o caminho absoluto da pasta raiz onde este script está rodando
REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
LOGS_DIR = REPO_ROOT / "logs"

# =====================================================================
# 2. CARREGAMENTO DAS VARIÁVEIS DE AMBIENTE (.env)
# =====================================================================
# Carrega o .env que está na mesma pasta do script
load_dotenv(REPO_ROOT / ".env") 

# Agora puxamos o Token e o Chat ID corretamente
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "SEU_TOKEN_AQUI")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "SEU_CHAT_ID_AQUI")

# Variável global para rastrear o processo do bot de trade
trade_process = None


# =====================================================================
# 3. FUNÇÕES AUXILIARES
# =====================================================================
def verificar_autorizacao(update: Update) -> bool:
    """Garante que apenas você (seu CHAT_ID) pode controlar o bot."""
    return str(update.effective_chat.id) == str(CHAT_ID)

def carregar_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def salvar_config(config_data):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, sort_keys=False, allow_unicode=True)

def ler_status_runtime():
    """Lê o arquivo de estado gerado pelo RuntimeGuard (o mesmo usado pelo launcher.py)"""
    # Procura o arquivo de estado mais recente na pasta logs
    arquivos_estado = list(LOGS_DIR.glob("state_mainnet_*.json"))
    if not arquivos_estado:
        return None
    
    # Pega o arquivo modificado mais recentemente
    arquivo_recente = max(arquivos_estado, key=os.path.getmtime)
    try:
        with open(arquivo_recente, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# =====================================================================
# 4. COMANDOS DO BOT (START, OFF, STATUS, ETC)
# =====================================================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Inicia as operações na Mainnet."""
    if not verificar_autorizacao(update): return
    
    global trade_process
    if trade_process is not None and trade_process.poll() is None:
        await update.message.reply_text("O Bot já está rodando!")
        return

    await update.message.reply_text("Iniciando o TraderBot na MAINNET com live trading ativado...")
    
    # Ele executa o trade bot como um subprocesso dentro do mesmo container
    cmd = [
        "python", "-m", "traderbot.main",
        "--config", str(CONFIG_PATH),
        "--network-override", "mainnet",
        "--execution-mode-override", "exchange",
        "--allow-live-trading",
        "run"
    ]
    
    trade_process = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
    await update.message.reply_text("Bot iniciado com sucesso em background!")

async def cmd_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Para as operações do bot."""
    if not verificar_autorizacao(update): return
    
    global trade_process
    if trade_process is None or trade_process.poll() is not None:
        await update.message.reply_text("O bot já está desligado.")
        return

    await update.message.reply_text("Desligando o bot... (Fechando processos)")
    trade_process.terminate()
    trade_process.wait()
    trade_process = None
    
    await update.message.reply_text("ot encerrado.")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra o status detalhado, IA e filtros."""
    if not verificar_autorizacao(update): return
    
    global trade_process
    is_running = trade_process is not None and trade_process.poll() is None
    
    estado = ler_status_runtime()
    
    if not is_running:
        await update.message.reply_text("*Status:* Bot DESLIGADO.", parse_mode="Markdown")
        return
    elif estado is None or not estado.get("last_cycle_payload"):
        await update.message.reply_text("*Status:* Bot LIGADO, aguardando o primeiro ciclo do mercado...", parse_mode="Markdown")
        return

    # Extrai os sub-dicionários do payload rico gerado pelo main.py
    payload = estado.get("last_cycle_payload", {})
    decision = payload.get("decision", {})
    filters = payload.get("filters", {})
    execution = payload.get("execution", {})

    # Status da Posição
    pos_label = execution.get("position_label", payload.get("position_label", "FLAT"))
    pnl = execution.get("position_unrealized_pnl", payload.get("position_unrealized_pnl", 0.0))
    entry_price = execution.get("position_avg_entry_price", payload.get("position_avg_entry_price", 0.0))
    bloqueio = filters.get("blocked_reason_human", payload.get("blocked_reason_human", None))

    msg = f"*STATUS DO BOT LIGADO*\n\n"
    msg += f"*Posição Atual:* {pos_label}\n"
    if pos_label != "FLAT":
        msg += f"*PnL Aberto:* ${pnl:.2f}\n"
        msg += f"*Preço Médio:* ${entry_price:.4f}\n"

    # Decisão da IA e Votação
    msg += f"\n*DECISÃO DA IA (Ensemble):*\n"
    msg += f"• Motivo: {decision.get('reason', payload.get('decision_reason', 'Avaliando...'))}\n"
    
    votes = decision.get("votes", payload.get("votes", {}))
    if votes:
        msg += f"• Placar: 🟢 {votes.get('buy', 0)} Compra | 🟡 {votes.get('hold', 0)} Neutro | 🔴 {votes.get('sell', 0)} Venda\n"

    model_votes = decision.get("model_votes", payload.get("model_votes", {}))
    if model_votes:
        msg += "• Votação Individual:\n"
        for model, vote in model_votes.items():
            emoji = "🟢" if vote > 0 else "🔴" if vote < 0 else "🟡"
            msg += f"  └ {model}: {vote:.2f} {emoji}\n"

    # Status dos Filtros de Proteção
    msg += f"\n*FILTROS DE ENTRADA:*\n"
    regime_valid = "Passou" if filters.get('regime_valid_for_entry', payload.get('regime_valid')) else "Reprovado"
    msg += f"• Regime Geral: {regime_valid}\n"
    msg += f"• Distância EMA 240: {filters.get('regime_dist_ema_240', payload.get('regime_dist_ema_240', 0.0)):.4f}\n"
    msg += f"• Volatilidade Z-Score: {filters.get('regime_vol_regime_z', payload.get('regime_vol_regime_z', 0.0)):.2f}\n"

    # Se houver algum bloqueio ativo impedindo trades
    if bloqueio:
        msg += f"\n🚧 *BLOQUEIO ATUAL:*\n{bloqueio}\n"

    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_dash(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra os dados de capital, risco e dimensionamento."""
    if not verificar_autorizacao(update): return
    
    estado = ler_status_runtime()
    if not estado or not estado.get("last_cycle_payload"):
        await update.message.reply_text("Ainda não há dados suficientes no ciclo atual para gerar o dashboard.")
        return

    payload = estado.get("last_cycle_payload", {})
    sizing = payload.get("sizing", {})
    market = payload.get("market_snapshot", {})

    # Extraindo dados financeiros
    saldo = sizing.get("sizing_balance_used", payload.get("sizing_balance_used", 0.0))
    disp = sizing.get("available_to_trade", payload.get("available_to_trade", 0.0))
    risk_pct = sizing.get("risk_pct", payload.get("risk_pct", 0.0))
    risk_amt = sizing.get("risk_amount", payload.get("risk_amount", 0.0))
    vol = sizing.get("adjusted_volume", payload.get("adjusted_volume", 0.0))
    
    ref_price = market.get("reference_price", payload.get("reference_price", 0.0))

    msg = f"*DASHBOARD FINANCEIRO E RISCO*\n\n"
    msg += f"*Capital Base Utilizado:* ${saldo:.2f}\n"
    msg += f"*Disponível na Corretora:* ${disp:.2f}\n\n"
    
    msg += f"*Gerenciamento de Risco (Próxima Trade):*\n"
    msg += f"• Risco Máximo: {risk_pct * 100:.2f}%\n"
    msg += f"• Exposição (Stop-Loss): ${risk_amt:.2f}\n"
    msg += f"• Contratos Calculados: {vol}\n\n"
    
    msg += f"*Mercado (Símbolo Atual):*\n"
    msg += f"• Preço Atual (Tick): ${ref_price:.4f}\n"

    await update.message.reply_text(msg, parse_mode="Markdown")
async def cmd_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Altera configurações rapidamente."""
    if not verificar_autorizacao(update): return
    
    args = context.args
    if len(args) == 0:
        # Mostra config atual
        cfg = carregar_config().get("environment", {})
        msg = "*Configurações Atuais:*\n\n"
        msg += f"1️⃣ Risco Máx: {cfg.get('max_risk_per_trade', 0.0)*100}%\n"
        msg += f"2️⃣ Stop Loss: {cfg.get('stop_loss_pct', 0.0)*100}%\n"
        msg += f"3️⃣ Take Profit: {cfg.get('take_profit_pct', 0.0)*100}%\n"
        msg += f"4️⃣ Filtro Hold: {cfg.get('action_hold_threshold', 0.0)}\n\n"
        msg += "Para alterar, use:\n`/config [risco|stop|tp|hold] [valor]`\nExemplo: `/config risco 1.5`"
        await update.message.reply_text(msg, parse_mode="Markdown")
        return

    if len(args) < 2:
        await update.message.reply_text("Formato incorreto. Use: `/config [parametro] [valor]`", parse_mode="Markdown")
        return

    parametro, valor_str = args[0].lower(), args[1]
    
    try:
        valor = float(valor_str)
    except ValueError:
        await update.message.reply_text("O valor deve ser um número válido.")
        return

    cfg_data = carregar_config()
    
    if parametro == "risco":
        cfg_data["environment"]["max_risk_per_trade"] = valor / 100.0
    elif parametro == "stop":
        cfg_data["environment"]["stop_loss_pct"] = valor / 100.0
    elif parametro == "tp":
        cfg_data["environment"]["take_profit_pct"] = valor / 100.0
    elif parametro == "hold":
        cfg_data["environment"]["action_hold_threshold"] = valor
    else:
        await update.message.reply_text("Parâmetro desconhecido. Escolha: risco, stop, tp ou hold.")
        return

    salvar_config(cfg_data)
    await update.message.reply_text(f"*{parametro.upper()}* alterado para {valor} com sucesso!\n_Obs: Se o bot estiver rodando, ele pegará a configuração no próximo ciclo._", parse_mode="Markdown")


# =====================================================================
# 5. RELATÓRIOS AUTOMÁTICOS
# =====================================================================
async def relatorio_diario(context: ContextTypes.DEFAULT_TYPE):
    """Executado automaticamente no fim do dia."""
    msg = f"*RELATÓRIO DIÁRIO - {date.today().strftime('%d/%m/%Y')}*\n\n"
    msg += "Fechamento do dia. O bot finalizou suas operações diárias.\n"
    await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")

async def relatorio_mensal(context: ContextTypes.DEFAULT_TYPE):
    """Executado todo dia 1º do mês."""
    if datetime.now().day != 1:
        return
    msg = f"🗓 *FECHAMENTO MENSAL*\n\nResumo do mês que passou...\n"
    await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")


# =====================================================================
# 6. INICIALIZAÇÃO DO BOT
# =====================================================================
def main():
    if TOKEN == "SEU_TOKEN_AQUI" or CHAT_ID == "SEU_CHAT_ID_AQUI" or not TOKEN:
        print("ALERTA: Configure o TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID no seu arquivo .env!")
        return

    app = Application.builder().token(TOKEN).build()

    # Registra os Comandos
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("off", cmd_off))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("dash", cmd_dash))
    app.add_handler(CommandHandler("config", cmd_config))

    # Agendamento de Relatórios
    app.job_queue.run_daily(relatorio_diario, time(hour=23, minute=55))
    app.job_queue.run_daily(relatorio_mensal, time(hour=9, minute=0))

    print("Painel do Telegram Iniciado com sucesso! Aguardando comandos...")
    app.run_polling()

if __name__ == "__main__":
    main()