# -*- coding: utf-8 -*-
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
REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
LOGS_DIR = REPO_ROOT / "logs"

# =====================================================================
# 2. CARREGAMENTO DAS VARIÁVEIS DE AMBIENTE (.env)
# =====================================================================
load_dotenv(REPO_ROOT / ".env") 

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "SEU_TOKEN_AQUI")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "SEU_CHAT_ID_AQUI")

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
    """Lê o arquivo de estado gerado pelo RuntimeGuard"""
    arquivos_estado = list(LOGS_DIR.glob("state_mainnet_*.json"))
    if not arquivos_estado:
        return None
    
    arquivo_recente = max(arquivos_estado, key=os.path.getmtime)
    try:
        with open(arquivo_recente, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def v_num(valor, default=0.0):
    """Garante que valores nulos sejam tratados como 0.0 para evitar erros no bot."""
    return float(valor) if valor is not None else default

def calcular_estatisticas_diarias():
    """Varre os arquivos de LOG reais para contar todo o histórico do dia."""
    hoje = date.today().isoformat()
    stats = {"abertas": 0, "fechadas": 0, "bloqueadas": 0, "vitorias": 0, "derrotas": 0}
    
    for arq in LOGS_DIR.glob("*.log"):
        try:
            with open(arq, "r", encoding="utf-8") as f:
                for linha in f:
                    if hoje in linha and "Ciclo runtime HL |" in linha:
                        try:
                            json_str = linha.split("Ciclo runtime HL |")[1].strip()
                            payload = json.loads(json_str)
                            
                            if payload.get("blocked_reason"): stats["bloqueadas"] += 1
                            if payload.get("opened_trade"): stats["abertas"] += 1
                            
                            if payload.get("closed_trade"):
                                stats["fechadas"] += 1
                                pnl = v_num(payload.get("position_unrealized_pnl"))
                                if pnl > 0: stats["vitorias"] += 1
                                else: stats["derrotas"] += 1
                        except: pass
        except: continue
        
    total_encerradas = stats["vitorias"] + stats["derrotas"]
    stats["winrate"] = (stats["vitorias"] / total_encerradas * 100) if total_encerradas > 0 else 0.0
    return stats


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
    
    await update.message.reply_text("Bot encerrado.")
    
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mostra o status detalhado, IA e filtros."""
    if not verificar_autorizacao(update): return
    
    global trade_process
    is_running = trade_process is not None and trade_process.poll() is None
    estado = ler_status_runtime()
    
    if not is_running:
        await update.message.reply_text("🔴 *STATUS:* Bot DESLIGADO.", parse_mode="Markdown")
        return
    if not estado or not estado.get("last_cycle_payload"):
        await update.message.reply_text("🟢 *STATUS:* Bot LIGADO.\n\n⏳ _Aguardando o fechamento do primeiro ciclo de mercado..._", parse_mode="Markdown")
        return

    payload = estado.get("last_cycle_payload", {})
    decision = payload.get("decision", {}) or {}
    filters = payload.get("filters", {}) or {}
    exec_info = payload.get("execution", {}) or {}

    msg = f"🛰️ *MONITORAMENTO EM TEMPO REAL*\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
    
    # --- 1. Estado da Posição ---
    pos_label = exec_info.get("position_label") or payload.get("position_label", "FLAT")
    if pos_label == "FLAT":
        msg += f"📊 *Estado da Posição:* FLAT (Sem posições abertas)\n\n"
    else:
        pnl = v_num(exec_info.get("position_unrealized_pnl") or payload.get("position_unrealized_pnl"))
        msg += f"📊 *Estado da Posição:* {pos_label}\n"
        msg += f"💰 *PnL Atual:* ${pnl:.2f}\n\n"

    # --- 2. Decisão da IA ---
    bucket_raw = str(decision.get("vote_bucket") or payload.get("vote_bucket") or "HOLD").upper()
    if bucket_raw == "HOLD": dec_str = "AGUARDAR"
    elif bucket_raw == "BUY": dec_str = "COMPRAR"
    elif bucket_raw == "SELL": dec_str = "VENDER"
    else: dec_str = bucket_raw

    conf = v_num(decision.get("confidence_pct") or payload.get("confidence_pct"))
    votes = decision.get("votes") or payload.get("votes", {})
    
    v_buy = votes.get('buy', 0) if isinstance(votes, dict) else 0
    v_hold = votes.get('hold', 0) if isinstance(votes, dict) else 0
    v_sell = votes.get('sell', 0) if isinstance(votes, dict) else 0

    msg += f"🧠 *DECISÃO DA IA (Ensemble):*\n"
    msg += f"• Decisão Final: {dec_str} (Força: {conf:.1f}%)\n"
    msg += f"• Placar: 🟢 {v_buy} Compra | 🟡 {v_hold} Neutro | 🔴 {v_sell} Venda\n\n"

    # --- 3. Filtros ---
    regime_ok = filters.get("regime_valid_for_entry", payload.get("regime_valid"))
    regime_str = "✅ Regime de Mercado (Permitido)" if regime_ok else "❌ Regime de Mercado (Bloqueado)"
    vol_z = v_num(filters.get("regime_vol_regime_z") or payload.get("regime_vol_regime_z"))
    dist_ema = v_num(filters.get("regime_dist_ema_240") or payload.get("regime_dist_ema_240"))

    msg += f"🛡️ *CHECK DE FILTROS:*\n"
    msg += f"{regime_str}\n"
    msg += f"  └ Volatilidade Z: {vol_z:.2f}\n"
    msg += f"  └ Dist. EMA 240: {dist_ema:.4f}\n"

    bloqueio = filters.get("blocked_reason_human") or payload.get("blocked_reason_human")
    if bloqueio:
        msg += f"\n🚧 *MOTIVO DO BLOQUEIO:*\n{bloqueio}\n"

    await update.message.reply_text(msg, parse_mode="Markdown")

async def cmd_dash(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Roda o check-hyperliquid internamente e mostra os dados financeiros reais."""
    if not verificar_autorizacao(update): return
    
    msg_loading = await update.message.reply_text("⏳ *Consultando Exchange e rodando cálculos...*", parse_mode="Markdown")

    try:
        cmd = [
            "python", "-m", "traderbot.main",
            "--config", str(CONFIG_PATH),
            "--network-override", "mainnet",
            "--execution-mode-override", "exchange",
            "check-hyperliquid"
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(REPO_ROOT)
        )
        stdout, stderr = await process.communicate()
        output = stdout.decode('utf-8')

        json_str = None
        for line in output.split('\n'):
            if "Status Hyperliquid: " in line:
                json_str = line.split("Status Hyperliquid: ")[1].strip()
                break

        if not json_str:
            await msg_loading.edit_text("❌ Erro ao ler dados da exchange. Verifique a conexão.")
            return

        hl_status = json.loads(json_str)

    except Exception as e:
        await msg_loading.edit_text(f"❌ Erro na consulta do check: {e}")
        return

    account = hl_status.get("account_summary", {})
    sizing = hl_status.get("validation_sizing", {})
    risk_limits = hl_status.get("risk_limits", {})
    
    saldo = v_num(account.get("sizing_balance_used"))
    disp = v_num(account.get("available_to_trade"))
    risco = v_num(risk_limits.get("max_risk_per_trade")) * 100
    expo = v_num(sizing.get("risk_amount"))
    contratos = v_num(sizing.get("adjusted_volume"))

    estado = ler_status_runtime()
    ref_price = 0.0
    if estado and estado.get("last_cycle_payload"):
        ref_price = v_num(estado["last_cycle_payload"].get("market_snapshot", {}).get("reference_price"))
    
    stats = calcular_estatisticas_diarias()
    hoje_str = datetime.now().strftime("%d/%m")

    msg = f"📈 *DASHBOARD DE PERFORMANCE*\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━\n"
    
    msg += f"💵 *Capital Base:* ${saldo:.2f}\n"
    msg += f"🏦 *Disponível:* ${disp:.2f}\n"
    msg += f"⚠️ *Risco Configurado:* {risco:.1f}%\n"
    msg += f"📏 *Exposição Calculada:* ${expo:.2f}\n"
    msg += f"⚖️ *Contratos Estimados:* {contratos}\n\n"

    msg += f"📅 *RESUMO DE HOJE ({hoje_str}):*\n"
    msg += f"✅ Abertas: {stats['abertas']}\n"
    msg += f"🧱 Bloqueadas: {stats['bloqueadas']}\n"
    msg += f"🏁 Encerradas: {stats['fechadas']}\n"
    msg += f"🎯 Winrate: {stats['winrate']:.1f}%\n\n"
    
    if ref_price > 0:
        msg += f"📊 *Mercado:* BTC em ${ref_price:.2f}"
    else:
        msg += f"📊 *Mercado:* Coletando dados de preço..."

    await msg_loading.edit_text(msg, parse_mode="Markdown")
    
    
async def cmd_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Altera configurações rapidamente."""
    if not verificar_autorizacao(update): return
    
    args = context.args
    if len(args) == 0:
        cfg = carregar_config().get("environment", {})
        msg = "*Configurações Atuais:*\n\n"
        msg += f"1️⃣ Risco Máx: {cfg.get('max_risk_per_trade', 0.0)*100:.1f}%\n"
        msg += f"2️⃣ Stop Loss: {cfg.get('stop_loss_pct', 0.0)*100:.1f}%\n"
        msg += f"3️⃣ Take Profit: {cfg.get('take_profit_pct', 0.0)*100:.1f}%\n"
        msg += f"4️⃣ Filtro Hold: {cfg.get('action_hold_threshold', 0.0)}\n"
        msg += f"5️⃣ Cooldown: {cfg.get('cooldown', 0)} candles\n\n"
        msg += "Para alterar, use:\n`/config [risco|stop|tp|hold|cooldown] [valor]`\nExemplo: `/config cooldown 3`"
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
    
    # Mapeamento dos parâmetros para o config.yaml
    if parametro == "risco":
        cfg_data["environment"]["max_risk_per_trade"] = valor / 100.0
    elif parametro == "stop":
        cfg_data["environment"]["stop_loss_pct"] = valor / 100.0
    elif parametro == "tp":
        cfg_data["environment"]["take_profit_pct"] = valor / 100.0
    elif parametro == "hold":
        cfg_data["environment"]["action_hold_threshold"] = valor
    elif parametro == "cooldown":
        # Cooldown geralmente é um número inteiro de candles
        cfg_data["environment"]["cooldown"] = int(valor) 
    else:
        await update.message.reply_text("Parâmetro desconhecido. Escolha: risco, stop, tp, hold ou cooldown.")
        return

    salvar_config(cfg_data)
    await update.message.reply_text(f"*{parametro.upper()}* alterado para {valor} com sucesso!\n_Obs: O bot pegará a nova configuração no próximo ciclo de mercado._", parse_mode="Markdown")


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
    msg = f"📊 *FECHAMENTO MENSAL*\n\nResumo do mês que passou...\n"
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