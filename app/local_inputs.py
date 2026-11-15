"""Permissão de interface baseada no bind do servidor, não no navegador."""


def local_uploads_enabled(server_address) -> bool:
    """Streamlit 1.32 usa server.address para definir o endereço de bind.

    Falha fechada para configuração ausente, wildcard, hostname ou IP externo.
    Não verifica túneis/proxies; o operador não deve expor esta execução local.
    """
    return isinstance(server_address, str) and server_address in {"127.0.0.1", "::1"}


LOCAL_COMMAND = "python -m streamlit run app/app.py --server.address=127.0.0.1"
