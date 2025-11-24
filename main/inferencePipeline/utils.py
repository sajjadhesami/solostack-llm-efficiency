from typing import Iterable, List, TypeVar


T = TypeVar("T")


def chunk_iter(items: Iterable[T], chunk_size: int) -> Iterable[List[T]]:
    batch: List[T] = []
    for it in items:
        batch.append(it)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


