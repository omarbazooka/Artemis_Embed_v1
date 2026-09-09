-- Harden the Artemis document-match RPC for server-side use only.

alter function public.match_document_chunks(extensions.vector, integer, uuid)
  set search_path = public, extensions;

revoke execute on function public.match_document_chunks(extensions.vector, integer, uuid) from public;
revoke execute on function public.match_document_chunks(extensions.vector, integer, uuid) from anon;
revoke execute on function public.match_document_chunks(extensions.vector, integer, uuid) from authenticated;
grant execute on function public.match_document_chunks(extensions.vector, integer, uuid) to service_role;
