from fastapi import Depends, Header, HTTPException
from supabase import Client

from app.core.db import db


def admin_user(authorization: str | None = Header(default=None), client: Client = Depends(db)) -> str:
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(401, 'Autenticação necessária')
    token = authorization.removeprefix('Bearer ').strip()
    try:
        user = client.auth.get_user(token).user
        if not user:
            raise ValueError('Token inválido')
        role = client.table('admin_users').select('user_id').eq('user_id', str(user.id)).limit(1).execute().data
        if not role:
            raise HTTPException(403, 'Acesso administrativo necessário')
        return str(user.id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(401, 'Sessão inválida') from exc

