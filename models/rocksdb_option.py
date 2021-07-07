import random
from configparser import ConfigParser

KB = 1024
MB = 1024 * 1024

internal_params = ['index','rocksdb.block.cache.miss COUNT', 'rocksdb.block.cache.hit COUNT', 'rocksdb.block.cache.add COUNT', 'rocksdb.block.cache.add.failures COUNT', 'rocksdb.block.cache.index.miss COUNT', 'rocksdb.block.cache.index.hit COUNT', 'rocksdb.block.cache.index.add COUNT', 'rocksdb.block.cache.index.bytes.insert COUNT', 'rocksdb.block.cache.index.bytes.evict COUNT', 'rocksdb.block.cache.filter.miss COUNT', 'rocksdb.block.cache.filter.hit COUNT', 'rocksdb.block.cache.filter.add COUNT', 'rocksdb.block.cache.filter.bytes.insert COUNT', 'rocksdb.block.cache.filter.bytes.evict COUNT', 'rocksdb.block.cache.data.miss COUNT', 'rocksdb.block.cache.data.hit COUNT', 'rocksdb.block.cache.data.add COUNT', 'rocksdb.block.cache.data.bytes.insert COUNT', 'rocksdb.block.cache.bytes.read COUNT', 'rocksdb.block.cache.bytes.write COUNT', 'rocksdb.bloom.filter.useful COUNT', 'rocksdb.bloom.filter.full.positive COUNT', 'rocksdb.bloom.filter.full.true.positive COUNT', 'rocksdb.bloom.filter.micros COUNT', 'rocksdb.persistent.cache.hit COUNT', 'rocksdb.persistent.cache.miss COUNT', 'rocksdb.sim.block.cache.hit COUNT', 'rocksdb.sim.block.cache.miss COUNT', 'rocksdb.memtable.hit COUNT', 'rocksdb.memtable.miss COUNT', 'rocksdb.l0.hit COUNT', 'rocksdb.l1.hit COUNT', 'rocksdb.l2andup.hit COUNT', 'rocksdb.compaction.key.drop.new COUNT', 'rocksdb.compaction.key.drop.obsolete COUNT', 'rocksdb.compaction.key.drop.range_del COUNT', 'rocksdb.compaction.key.drop.user COUNT', 'rocksdb.compaction.range_del.drop.obsolete COUNT', 'rocksdb.compaction.optimized.del.drop.obsolete COUNT', 'rocksdb.compaction.cancelled COUNT', 'rocksdb.number.keys.written COUNT', 'rocksdb.number.keys.read COUNT', 'rocksdb.number.keys.updated COUNT', 'rocksdb.bytes.written COUNT', 'rocksdb.bytes.read COUNT', 'rocksdb.number.db.seek COUNT', 'rocksdb.number.db.next COUNT', 'rocksdb.number.db.prev COUNT', 'rocksdb.number.db.seek.found COUNT', 'rocksdb.number.db.next.found COUNT', 'rocksdb.number.db.prev.found COUNT', 'rocksdb.db.iter.bytes.read COUNT', 'rocksdb.no.file.closes COUNT', 'rocksdb.no.file.opens COUNT', 'rocksdb.no.file.errors COUNT', 'rocksdb.l0.slowdown.micros COUNT', 'rocksdb.memtable.compaction.micros COUNT', 'rocksdb.l0.num.files.stall.micros COUNT', 'rocksdb.stall.micros COUNT', 'rocksdb.db.mutex.wait.micros COUNT', 'rocksdb.rate.limit.delay.millis COUNT', 'rocksdb.num.iterators COUNT', 'rocksdb.number.multiget.get COUNT', 'rocksdb.number.multiget.keys.read COUNT', 'rocksdb.number.multiget.bytes.read COUNT', 'rocksdb.number.deletes.filtered COUNT', 'rocksdb.number.merge.failures COUNT', 'rocksdb.bloom.filter.prefix.checked COUNT', 'rocksdb.bloom.filter.prefix.useful COUNT', 'rocksdb.number.reseeks.iteration COUNT', 'rocksdb.getupdatessince.calls COUNT', 'rocksdb.block.cachecompressed.miss COUNT', 'rocksdb.block.cachecompressed.hit COUNT', 'rocksdb.block.cachecompressed.add COUNT', 'rocksdb.block.cachecompressed.add.failures COUNT', 'rocksdb.wal.synced COUNT', 'rocksdb.wal.bytes COUNT', 'rocksdb.write.self COUNT', 'rocksdb.write.other COUNT', 'rocksdb.write.timeout COUNT', 'rocksdb.write.wal COUNT', 'rocksdb.compact.read.bytes COUNT', 'rocksdb.compact.write.bytes COUNT', 'rocksdb.flush.write.bytes COUNT', 'rocksdb.compact.read.marked.bytes COUNT', 'rocksdb.compact.read.periodic.bytes COUNT', 'rocksdb.compact.read.ttl.bytes COUNT', 'rocksdb.compact.write.marked.bytes COUNT', 'rocksdb.compact.write.periodic.bytes COUNT', 'rocksdb.compact.write.ttl.bytes COUNT', 'rocksdb.number.direct.load.table.properties COUNT', 'rocksdb.number.superversion_acquires COUNT', 'rocksdb.number.superversion_releases COUNT', 'rocksdb.number.superversion_cleanups COUNT', 'rocksdb.number.block.compressed COUNT', 'rocksdb.number.block.decompressed COUNT', 'rocksdb.number.block.not_compressed COUNT', 'rocksdb.merge.operation.time.nanos COUNT', 'rocksdb.filter.operation.time.nanos COUNT', 'rocksdb.row.cache.hit COUNT', 'rocksdb.row.cache.miss COUNT', 'rocksdb.read.amp.estimate.useful.bytes COUNT', 'rocksdb.read.amp.total.read.bytes COUNT', 'rocksdb.number.rate_limiter.drains COUNT', 'rocksdb.number.iter.skip COUNT', 'rocksdb.blobdb.num.put COUNT', 'rocksdb.blobdb.num.write COUNT', 'rocksdb.blobdb.num.get COUNT', 'rocksdb.blobdb.num.multiget COUNT', 'rocksdb.blobdb.num.seek COUNT', 'rocksdb.blobdb.num.next COUNT', 'rocksdb.blobdb.num.prev COUNT', 'rocksdb.blobdb.num.keys.written COUNT', 'rocksdb.blobdb.num.keys.read COUNT', 'rocksdb.blobdb.bytes.written COUNT', 'rocksdb.blobdb.bytes.read COUNT', 'rocksdb.blobdb.write.inlined COUNT', 'rocksdb.blobdb.write.inlined.ttl COUNT', 'rocksdb.blobdb.write.blob COUNT', 'rocksdb.blobdb.write.blob.ttl COUNT', 'rocksdb.blobdb.blob.file.bytes.written COUNT', 'rocksdb.blobdb.blob.file.bytes.read COUNT', 'rocksdb.blobdb.blob.file.synced COUNT', 'rocksdb.blobdb.blob.index.expired.count COUNT', 'rocksdb.blobdb.blob.index.expired.size COUNT', 'rocksdb.blobdb.blob.index.evicted.count COUNT', 'rocksdb.blobdb.blob.index.evicted.size COUNT', 'rocksdb.blobdb.gc.num.files COUNT', 'rocksdb.blobdb.gc.num.new.files COUNT', 'rocksdb.blobdb.gc.failures COUNT', 'rocksdb.blobdb.gc.num.keys.overwritten COUNT', 'rocksdb.blobdb.gc.num.keys.expired COUNT', 'rocksdb.blobdb.gc.num.keys.relocated COUNT', 'rocksdb.blobdb.gc.bytes.overwritten COUNT', 'rocksdb.blobdb.gc.bytes.expired COUNT', 'rocksdb.blobdb.gc.bytes.relocated COUNT', 'rocksdb.blobdb.fifo.num.files.evicted COUNT', 'rocksdb.blobdb.fifo.num.keys.evicted COUNT', 'rocksdb.blobdb.fifo.bytes.evicted COUNT', 'rocksdb.txn.overhead.mutex.prepare COUNT', 'rocksdb.txn.overhead.mutex.old.commit.map COUNT', 'rocksdb.txn.overhead.duplicate.key COUNT', 'rocksdb.txn.overhead.mutex.snapshot COUNT', 'rocksdb.txn.get.tryagain COUNT', 'rocksdb.number.multiget.keys.found COUNT', 'rocksdb.num.iterator.created COUNT', 'rocksdb.num.iterator.deleted COUNT', 'rocksdb.block.cache.compression.dict.miss COUNT', 'rocksdb.block.cache.compression.dict.hit COUNT', 'rocksdb.block.cache.compression.dict.add COUNT', 'rocksdb.block.cache.compression.dict.bytes.insert COUNT', 'rocksdb.block.cache.compression.dict.bytes.evict COUNT', 'rocksdb.block.cache.add.redundant COUNT', 'rocksdb.block.cache.index.add.redundant COUNT', 'rocksdb.block.cache.filter.add.redundant COUNT', 'rocksdb.block.cache.data.add.redundant COUNT', 'rocksdb.block.cache.compression.dict.add.redundant COUNT', 'rocksdb.files.marked.trash COUNT', 'rocksdb.files.deleted.immediately COUNT', 'rocksdb.error.handler.bg.errro.count COUNT', 'rocksdb.error.handler.bg.io.errro.count COUNT', 'rocksdb.error.handler.bg.retryable.io.errro.count COUNT', 'rocksdb.error.handler.autoresume.count COUNT', 'rocksdb.error.handler.autoresume.retry.total.count COUNT', 'rocksdb.error.handler.autoresume.success.count COUNT', 'rocksdb.db.get.micros_P50', 'rocksdb.db.get.micros_P95', 'rocksdb.db.get.micros_P99', 'rocksdb.db.get.micros_P100', 'rocksdb.db.get.micros_COUNT', 'rocksdb.db.get.micros_SUM', 'rocksdb.db.write.micros_P50', 'rocksdb.db.write.micros_P95', 'rocksdb.db.write.micros_P99', 'rocksdb.db.write.micros_P100', 'rocksdb.db.write.micros_COUNT', 'rocksdb.db.write.micros_SUM', 'rocksdb.compaction.times.micros_P50', 'rocksdb.compaction.times.micros_P95', 'rocksdb.compaction.times.micros_P99', 'rocksdb.compaction.times.micros_P100', 'rocksdb.compaction.times.micros_COUNT', 'rocksdb.compaction.times.micros_SUM', 'rocksdb.compaction.times.cpu_micros_P50', 'rocksdb.compaction.times.cpu_micros_P95', 'rocksdb.compaction.times.cpu_micros_P99', 'rocksdb.compaction.times.cpu_micros_P100', 'rocksdb.compaction.times.cpu_micros_COUNT', 'rocksdb.compaction.times.cpu_micros_SUM', 'rocksdb.subcompaction.setup.times.micros_P50', 'rocksdb.subcompaction.setup.times.micros_P95', 'rocksdb.subcompaction.setup.times.micros_P99', 'rocksdb.subcompaction.setup.times.micros_P100', 'rocksdb.subcompaction.setup.times.micros_COUNT', 'rocksdb.subcompaction.setup.times.micros_SUM', 'rocksdb.table.sync.micros_P50', 'rocksdb.table.sync.micros_P95', 'rocksdb.table.sync.micros_P99', 'rocksdb.table.sync.micros_P100', 'rocksdb.table.sync.micros_COUNT', 'rocksdb.table.sync.micros_SUM', 'rocksdb.compaction.outfile.sync.micros_P50', 'rocksdb.compaction.outfile.sync.micros_P95', 'rocksdb.compaction.outfile.sync.micros_P99', 'rocksdb.compaction.outfile.sync.micros_P100', 'rocksdb.compaction.outfile.sync.micros_COUNT', 'rocksdb.compaction.outfile.sync.micros_SUM', 'rocksdb.wal.file.sync.micros_P50', 'rocksdb.wal.file.sync.micros_P95', 'rocksdb.wal.file.sync.micros_P99', 'rocksdb.wal.file.sync.micros_P100', 'rocksdb.wal.file.sync.micros_COUNT', 'rocksdb.wal.file.sync.micros_SUM', 'rocksdb.manifest.file.sync.micros_P50', 'rocksdb.manifest.file.sync.micros_P95', 'rocksdb.manifest.file.sync.micros_P99', 'rocksdb.manifest.file.sync.micros_P100', 'rocksdb.manifest.file.sync.micros_COUNT', 'rocksdb.manifest.file.sync.micros_SUM', 'rocksdb.table.open.io.micros_P50', 'rocksdb.table.open.io.micros_P95', 'rocksdb.table.open.io.micros_P99', 'rocksdb.table.open.io.micros_P100', 'rocksdb.table.open.io.micros_COUNT', 'rocksdb.table.open.io.micros_SUM', 'rocksdb.db.multiget.micros_P50', 'rocksdb.db.multiget.micros_P95', 'rocksdb.db.multiget.micros_P99', 'rocksdb.db.multiget.micros_P100', 'rocksdb.db.multiget.micros_COUNT', 'rocksdb.db.multiget.micros_SUM', 'rocksdb.read.block.compaction.micros_P50', 'rocksdb.read.block.compaction.micros_P95', 'rocksdb.read.block.compaction.micros_P99', 'rocksdb.read.block.compaction.micros_P100', 'rocksdb.read.block.compaction.micros_COUNT', 'rocksdb.read.block.compaction.micros_SUM', 'rocksdb.read.block.get.micros_P50', 'rocksdb.read.block.get.micros_P95', 'rocksdb.read.block.get.micros_P99', 'rocksdb.read.block.get.micros_P100', 'rocksdb.read.block.get.micros_COUNT', 'rocksdb.read.block.get.micros_SUM', 'rocksdb.write.raw.block.micros_P50', 'rocksdb.write.raw.block.micros_P95', 'rocksdb.write.raw.block.micros_P99', 'rocksdb.write.raw.block.micros_P100', 'rocksdb.write.raw.block.micros_COUNT', 'rocksdb.write.raw.block.micros_SUM', 'rocksdb.l0.slowdown.count_P50', 'rocksdb.l0.slowdown.count_P95', 'rocksdb.l0.slowdown.count_P99', 'rocksdb.l0.slowdown.count_P100', 'rocksdb.l0.slowdown.count_COUNT', 'rocksdb.l0.slowdown.count_SUM', 'rocksdb.memtable.compaction.count_P50', 'rocksdb.memtable.compaction.count_P95', 'rocksdb.memtable.compaction.count_P99', 'rocksdb.memtable.compaction.count_P100', 'rocksdb.memtable.compaction.count_COUNT', 'rocksdb.memtable.compaction.count_SUM', 'rocksdb.num.files.stall.count_P50', 'rocksdb.num.files.stall.count_P95', 'rocksdb.num.files.stall.count_P99', 'rocksdb.num.files.stall.count_P100', 'rocksdb.num.files.stall.count_COUNT', 'rocksdb.num.files.stall.count_SUM', 'rocksdb.hard.rate.limit.delay.count_P50', 'rocksdb.hard.rate.limit.delay.count_P95', 'rocksdb.hard.rate.limit.delay.count_P99', 'rocksdb.hard.rate.limit.delay.count_P100', 'rocksdb.hard.rate.limit.delay.count_COUNT', 'rocksdb.hard.rate.limit.delay.count_SUM', 'rocksdb.soft.rate.limit.delay.count_P50', 'rocksdb.soft.rate.limit.delay.count_P95', 'rocksdb.soft.rate.limit.delay.count_P99', 'rocksdb.soft.rate.limit.delay.count_P100', 'rocksdb.soft.rate.limit.delay.count_COUNT', 'rocksdb.soft.rate.limit.delay.count_SUM', 'rocksdb.numfiles.in.singlecompaction_P50', 'rocksdb.numfiles.in.singlecompaction_P95', 'rocksdb.numfiles.in.singlecompaction_P99', 'rocksdb.numfiles.in.singlecompaction_P100', 'rocksdb.numfiles.in.singlecompaction_COUNT', 'rocksdb.numfiles.in.singlecompaction_SUM', 'rocksdb.db.seek.micros_P50', 'rocksdb.db.seek.micros_P95', 'rocksdb.db.seek.micros_P99', 'rocksdb.db.seek.micros_P100', 'rocksdb.db.seek.micros_COUNT', 'rocksdb.db.seek.micros_SUM', 'rocksdb.db.write.stall_P50', 'rocksdb.db.write.stall_P95', 'rocksdb.db.write.stall_P99', 'rocksdb.db.write.stall_P100', 'rocksdb.db.write.stall_COUNT', 'rocksdb.db.write.stall_SUM', 'rocksdb.sst.read.micros_P50', 'rocksdb.sst.read.micros_P95', 'rocksdb.sst.read.micros_P99', 'rocksdb.sst.read.micros_P100', 'rocksdb.sst.read.micros_COUNT', 'rocksdb.sst.read.micros_SUM', 'rocksdb.num.subcompactions.scheduled_P50', 'rocksdb.num.subcompactions.scheduled_P95', 'rocksdb.num.subcompactions.scheduled_P99', 'rocksdb.num.subcompactions.scheduled_P100', 'rocksdb.num.subcompactions.scheduled_COUNT', 'rocksdb.num.subcompactions.scheduled_SUM', 'rocksdb.bytes.per.read_P50', 'rocksdb.bytes.per.read_P95', 'rocksdb.bytes.per.read_P99', 'rocksdb.bytes.per.read_P100', 'rocksdb.bytes.per.read_COUNT', 'rocksdb.bytes.per.read_SUM', 'rocksdb.bytes.per.write_P50', 'rocksdb.bytes.per.write_P95', 'rocksdb.bytes.per.write_P99', 'rocksdb.bytes.per.write_P100', 'rocksdb.bytes.per.write_COUNT', 'rocksdb.bytes.per.write_SUM', 'rocksdb.bytes.per.multiget_P50', 'rocksdb.bytes.per.multiget_P95', 'rocksdb.bytes.per.multiget_P99', 'rocksdb.bytes.per.multiget_P100', 'rocksdb.bytes.per.multiget_COUNT', 'rocksdb.bytes.per.multiget_SUM', 'rocksdb.bytes.compressed_P50', 'rocksdb.bytes.compressed_P95', 'rocksdb.bytes.compressed_P99', 'rocksdb.bytes.compressed_P100', 'rocksdb.bytes.compressed_COUNT', 'rocksdb.bytes.compressed_SUM', 'rocksdb.bytes.decompressed_P50', 'rocksdb.bytes.decompressed_P95', 'rocksdb.bytes.decompressed_P99', 'rocksdb.bytes.decompressed_P100', 'rocksdb.bytes.decompressed_COUNT', 'rocksdb.bytes.decompressed_SUM', 'rocksdb.compression.times.nanos_P50', 'rocksdb.compression.times.nanos_P95', 'rocksdb.compression.times.nanos_P99', 'rocksdb.compression.times.nanos_P100', 'rocksdb.compression.times.nanos_COUNT', 'rocksdb.compression.times.nanos_SUM', 'rocksdb.decompression.times.nanos_P50', 'rocksdb.decompression.times.nanos_P95', 'rocksdb.decompression.times.nanos_P99', 'rocksdb.decompression.times.nanos_P100', 'rocksdb.decompression.times.nanos_COUNT', 'rocksdb.decompression.times.nanos_SUM', 'rocksdb.read.num.merge_operands_P50', 'rocksdb.read.num.merge_operands_P95', 'rocksdb.read.num.merge_operands_P99', 'rocksdb.read.num.merge_operands_P100', 'rocksdb.read.num.merge_operands_COUNT', 'rocksdb.read.num.merge_operands_SUM', 'rocksdb.blobdb.key.size_P50', 'rocksdb.blobdb.key.size_P95', 'rocksdb.blobdb.key.size_P99', 'rocksdb.blobdb.key.size_P100', 'rocksdb.blobdb.key.size_COUNT', 'rocksdb.blobdb.key.size_SUM', 'rocksdb.blobdb.value.size_P50', 'rocksdb.blobdb.value.size_P95', 'rocksdb.blobdb.value.size_P99', 'rocksdb.blobdb.value.size_P100', 'rocksdb.blobdb.value.size_COUNT', 'rocksdb.blobdb.value.size_SUM', 'rocksdb.blobdb.write.micros_P50', 'rocksdb.blobdb.write.micros_P95', 'rocksdb.blobdb.write.micros_P99', 'rocksdb.blobdb.write.micros_P100', 'rocksdb.blobdb.write.micros_COUNT', 'rocksdb.blobdb.write.micros_SUM', 'rocksdb.blobdb.get.micros_P50', 'rocksdb.blobdb.get.micros_P95', 'rocksdb.blobdb.get.micros_P99', 'rocksdb.blobdb.get.micros_P100', 'rocksdb.blobdb.get.micros_COUNT', 'rocksdb.blobdb.get.micros_SUM', 'rocksdb.blobdb.multiget.micros_P50', 'rocksdb.blobdb.multiget.micros_P95', 'rocksdb.blobdb.multiget.micros_P99', 'rocksdb.blobdb.multiget.micros_P100', 'rocksdb.blobdb.multiget.micros_COUNT', 'rocksdb.blobdb.multiget.micros_SUM', 'rocksdb.blobdb.seek.micros_P50', 'rocksdb.blobdb.seek.micros_P95', 'rocksdb.blobdb.seek.micros_P99', 'rocksdb.blobdb.seek.micros_P100', 'rocksdb.blobdb.seek.micros_COUNT', 'rocksdb.blobdb.seek.micros_SUM', 'rocksdb.blobdb.next.micros_P50', 'rocksdb.blobdb.next.micros_P95', 'rocksdb.blobdb.next.micros_P99', 'rocksdb.blobdb.next.micros_P100', 'rocksdb.blobdb.next.micros_COUNT', 'rocksdb.blobdb.next.micros_SUM', 'rocksdb.blobdb.prev.micros_P50', 'rocksdb.blobdb.prev.micros_P95', 'rocksdb.blobdb.prev.micros_P99', 'rocksdb.blobdb.prev.micros_P100', 'rocksdb.blobdb.prev.micros_COUNT', 'rocksdb.blobdb.prev.micros_SUM', 'rocksdb.blobdb.blob.file.write.micros_P50', 'rocksdb.blobdb.blob.file.write.micros_P95', 'rocksdb.blobdb.blob.file.write.micros_P99', 'rocksdb.blobdb.blob.file.write.micros_P100', 'rocksdb.blobdb.blob.file.write.micros_COUNT', 'rocksdb.blobdb.blob.file.write.micros_SUM', 'rocksdb.blobdb.blob.file.read.micros_P50', 'rocksdb.blobdb.blob.file.read.micros_P95', 'rocksdb.blobdb.blob.file.read.micros_P99', 'rocksdb.blobdb.blob.file.read.micros_P100', 'rocksdb.blobdb.blob.file.read.micros_COUNT', 'rocksdb.blobdb.blob.file.read.micros_SUM', 'rocksdb.blobdb.blob.file.sync.micros_P50', 'rocksdb.blobdb.blob.file.sync.micros_P95', 'rocksdb.blobdb.blob.file.sync.micros_P99', 'rocksdb.blobdb.blob.file.sync.micros_P100', 'rocksdb.blobdb.blob.file.sync.micros_COUNT', 'rocksdb.blobdb.blob.file.sync.micros_SUM', 'rocksdb.blobdb.gc.micros_P50', 'rocksdb.blobdb.gc.micros_P95', 'rocksdb.blobdb.gc.micros_P99', 'rocksdb.blobdb.gc.micros_P100', 'rocksdb.blobdb.gc.micros_COUNT', 'rocksdb.blobdb.gc.micros_SUM', 'rocksdb.blobdb.compression.micros_P50', 'rocksdb.blobdb.compression.micros_P95', 'rocksdb.blobdb.compression.micros_P99', 'rocksdb.blobdb.compression.micros_P100', 'rocksdb.blobdb.compression.micros_COUNT', 'rocksdb.blobdb.compression.micros_SUM', 'rocksdb.blobdb.decompression.micros_P50', 'rocksdb.blobdb.decompression.micros_P95', 'rocksdb.blobdb.decompression.micros_P99', 'rocksdb.blobdb.decompression.micros_P100', 'rocksdb.blobdb.decompression.micros_COUNT', 'rocksdb.blobdb.decompression.micros_SUM', 'rocksdb.db.flush.micros_P50', 'rocksdb.db.flush.micros_P95', 'rocksdb.db.flush.micros_P99', 'rocksdb.db.flush.micros_P100', 'rocksdb.db.flush.micros_COUNT', 'rocksdb.db.flush.micros_SUM', 'rocksdb.sst.batch.size_P50', 'rocksdb.sst.batch.size_P95', 'rocksdb.sst.batch.size_P99', 'rocksdb.sst.batch.size_P100', 'rocksdb.sst.batch.size_COUNT', 'rocksdb.sst.batch.size_SUM', 'rocksdb.num.index.and.filter.blocks.read.per.level_P50', 'rocksdb.num.index.and.filter.blocks.read.per.level_P95', 'rocksdb.num.index.and.filter.blocks.read.per.level_P99', 'rocksdb.num.index.and.filter.blocks.read.per.level_P100', 'rocksdb.num.index.and.filter.blocks.read.per.level_COUNT', 'rocksdb.num.index.and.filter.blocks.read.per.level_SUM', 'rocksdb.num.data.blocks.read.per.level_P50', 'rocksdb.num.data.blocks.read.per.level_P95', 'rocksdb.num.data.blocks.read.per.level_P99', 'rocksdb.num.data.blocks.read.per.level_P100', 'rocksdb.num.data.blocks.read.per.level_COUNT', 'rocksdb.num.data.blocks.read.per.level_SUM', 'rocksdb.num.sst.read.per.level_P50', 'rocksdb.num.sst.read.per.level_P95', 'rocksdb.num.sst.read.per.level_P99', 'rocksdb.num.sst.read.per.level_P100', 'rocksdb.num.sst.read.per.level_COUNT', 'rocksdb.num.sst.read.per.level_SUM', 'rocksdb.error.handler.autoresume.retry.count_P50', 'rocksdb.error.handler.autoresume.retry.count_P95', 'rocksdb.error.handler.autoresume.retry.count_P99', 'rocksdb.error.handler.autoresume.retry.count_P100', 'rocksdb.error.handler.autoresume.retry.count_COUNT', 'rocksdb.error.handler.autoresume.retry.count_SUM']
external_params = [
    "index",
    'key_size',
    'value_size',
    'num',
    'max_background_compactions', 
    'max_background_flushes', 
    'write_buffer_size', 
    'max_write_buffer_number', 
    'min_write_buffer_number_to_merge', 
    'compaction_pri', 
    'compaction_style', 
    'level0_file_num_compaction_trigger', 
    'level0_slowdown_writes_trigger', 
    'level0_stop_writes_trigger', 
    'compression_type', 
    'bloom_locality', 
    'open_files', 
    'block_size', 
    'cache_index_and_filter_blocks', 
    'max_bytes_for_level_base', 
    'max_bytes_for_level_multiplier', 
    'target_file_size_base', 
    'target_file_size_multiplier', 
    'num_levels', 
    'memtable_bloom_size_ratio', 
    'compression_ratio',
]
outputs = [
    "index",
    'TIME',
    'RATE',
    'WAF',
    'SA'
]

# Change parameter values from line 45 to line 88
option = {
    "max_background_compactions": [i for i in range(1, 17)], # D:1, B:4 ~ 32
    "max_background_flushes": [i for i in range(1, 17)], #D:1, B:4~32
    "write_buffer_size": [s * KB for s in range(512, 2048)], #D:64M, B:0.25M ~ 1M
    "max_write_buffer_number": [i for i in range(2, 9)], #D:2, B:2~16
    "min_write_buffer_number_to_merge": [i for i in range(1, 3)], #D:1
    "compaction_pri": { #D:0
        "kByCompensatedSize" : 0,
        "kOldestLargestSeqFirst" : 1,
        "kOldestSmallestSeqFirst" : 2,
        "kMinOverlappingRatio" : 3
    },
    "compaction_style": { #D:0
        "kCompactionStyleLevel" : 0, 
        "kCompactionStyleUniversal" : 1,
        "kCompactionStyleFIFO" : 2,
        "kCompactionStyleNone" : 3
    },
    "level0_file_num_compaction_trigger": [i for i in range(2, 9)], #D:4, B:2 ~ 8
    "level0_slowdown_writes_trigger": [i for i in range(16, 33)], #D:20, B:16 ~ 64
    "level0_stop_writes_trigger": [i for i in range(32, 65)], #D:36, B:64 ~ 128
    "compression_type": [i for i in range(0, 4)], #D:"snappy", B:no "bzip2"
    "bloom_locality": [0, 1], #D:0
    "open_files": [-1, 10000, 100000, 1000000], #D:-1 B:-1
    "block_size": [s * KB for s in range(2, 17)], #D:4096, B:4096 ~ 32768
    "cache_index_and_filter_blocks": [1, 0] #D:false
}

plus_option = {
    "memtable_bloom_size_ratio": [0, 0.05, 0.1, 0.15, 0.2], #D:0
    "compression_ratio": [i/100 for i in range(100)] #D:0.5, B:0.1 ~ 0.9
}

level_compaction_option = {
    "max_bytes_for_level_base": [s * MB for s in range(2, 9)], #D:256M, B:1M ~ 16M
    "max_bytes_for_level_multiplier": [i for i in range(8, 13)], #D:10, B:6 ~ 10
    "target_file_size_base": [s * KB for s in range(512, 2049)], #D:64M, B:0.25M ~ 4M
    "target_file_size_multiplier": [ i for i in range(1, 3)], #D:1, B:1 ~ 2
    "num_levels": [5, 6, 7, 8] #D:7, B:7
}

universal_compaction_option = {
    "universal_max_size_amplification_percent": [],
    "universal_size_ratio ": [],
    "universal_min_merge_width": [],
    "universal_max_size_amplification_percent": [],
    "universal_compression_size_percent": []
}

unknown_option = {
    "bytes_per_sync": [],
    "wal_bytes_per_sync": []
}

def read_config_option(config_file : str):
    parser = ConfigParser()
    parser.read(config_file)
    option_list = ""
    option_dict = {}

    for k in parser.options("rocksdb"):
        value = parser.get("rocksdb", k)
        opt = f"-{k}={value} "
        option_list += opt
        option_dict[k] = value

    return (option_list, option_dict)

    

def make_random_option():
    option_list = ""
    option_dict = {}

    compaction_style = ""
    write_buffer_number = -1

    # option
    for k, v in option.items():

        if k == "max_write_buffer_number":
            write_buffer_number = random.choice(v)
            value = write_buffer_number
            opt = f"-{k}={value} "
        elif k == "min_write_buffer_number_to_merge":
            lt_write_buffers = [i for i in v if i <= write_buffer_number]
            value = random.choice(lt_write_buffers)
            opt = f"-{k}={value} "
        elif k == "compaction_pri":
            value = random.choice(list(v.values()))
            opt = f"-{k}={value} "
        elif k == "compaction_style":
            # compaction_style = random.choice(list(v.keys()))
            compaction_style = "kCompactionStyleLevel"
            value = v[compaction_style]
            opt = f"-{k}={value} "
        else:
            value = random.choice(v)
            opt = f"-{k}={value} "

        option_list += opt
        option_dict[k] = value
    
    
    # compaction option
    if compaction_style == "kCompactionStyleLevel":
        for k, v in level_compaction_option.items():
            value = random.choice(v)
            opt = f"-{k}={value} "
            option_list += opt
            option_dict[k] = value  
    elif compaction_style == "kCompactionStyleUniversal":
        for k, v in universal_compaction_option.items():
            value = random.choice(v)
            opt = f"-{k}={value} "
            option_list += opt
            option_dict[k] = value
    else:
        pass


    # plus_option
    for k, v in plus_option.items():
        value = random.choice(v)
        opt = f"-{k}={value} "
        option_list += opt
        option_dict[k] = value
    
    return (option_list, option_dict)

def save_option_as_cnf(data:dict, filename:str):

    config = ConfigParser()
    config['rocksdb'] = data

    with open(filename, 'w') as f:
        config.write(f)
