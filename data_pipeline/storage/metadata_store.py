"""
PostgreSQL元数据管理
记录每条数据的来源、处理状态、质量分数、标签等
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import asyncio
import json

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """数据集信息"""
    dataset_id: int
    name: str
    description: str
    data_type: str  # text, image_text, audio, video
    total_items: int
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dataset_id': self.dataset_id,
            'name': self.name,
            'description': self.description,
            'data_type': self.data_type,
            'total_items': self.total_items,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class DataItemMeta:
    """数据项元数据"""
    item_id: str
    dataset_id: int
    data_type: str
    object_path: str
    status: str  # raw, cleaned, labeled, validated
    quality_score: Optional[float]
    labels: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'item_id': self.item_id,
            'dataset_id': self.dataset_id,
            'data_type': self.data_type,
            'object_path': self.object_path,
            'status': self.status,
            'quality_score': self.quality_score,
            'labels': self.labels,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class DatasetStats:
    """数据集统计信息"""
    total_items: int
    status_distribution: Dict[str, int]
    quality_distribution: Dict[str, float]
    data_type_distribution: Dict[str, int]
    avg_quality_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_items': self.total_items,
            'status_distribution': self.status_distribution,
            'quality_distribution': self.quality_distribution,
            'data_type_distribution': self.data_type_distribution,
            'avg_quality_score': self.avg_quality_score
        }


class MetadataStore:
    """
    基于PostgreSQL的数据集元数据管理。
    记录每条数据的来源、处理状态、质量分数、标签等。
    """
    
    def __init__(self, dsn: str):
        """
        初始化元数据存储
        
        Args:
            dsn: 数据库连接字符串 (如: postgresql://user:pass@localhost:5432/db)
        """
        self.dsn = dsn
        self.pool = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def connect(self):
        """连接数据库"""
        try:
            import asyncpg
            
            self.pool = await asyncpg.create_pool(self.dsn, min_size=5, max_size=20)
            self.logger.info(f"Connected to PostgreSQL: {self.dsn}")
            
            # 创建表结构
            await self._create_tables()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    async def close(self):
        """关闭数据库连接"""
        if self.pool:
            await self.pool.close()
            self.logger.info("PostgreSQL connection closed")
    
    async def _create_tables(self):
        """创建数据库表"""
        create_tables_sql = """
        -- 数据集表
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            description TEXT,
            data_type VARCHAR(50) NOT NULL,
            total_items INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        
        -- 数据项表
        CREATE TABLE IF NOT EXISTS data_items (
            item_id VARCHAR(255) PRIMARY KEY,
            dataset_id INTEGER REFERENCES datasets(dataset_id),
            data_type VARCHAR(50) NOT NULL,
            object_path TEXT NOT NULL,
            content_hash VARCHAR(64),
            status VARCHAR(50) DEFAULT 'raw',
            quality_score FLOAT,
            labels JSONB DEFAULT '{}'::jsonb,
            metadata JSONB DEFAULT '{}'::jsonb,
            source JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- 创建索引
        CREATE INDEX IF NOT EXISTS idx_data_items_dataset_id ON data_items(dataset_id);
        CREATE INDEX IF NOT EXISTS idx_data_items_status ON data_items(status);
        CREATE INDEX IF NOT EXISTS idx_data_items_data_type ON data_items(data_type);
        CREATE INDEX IF NOT EXISTS idx_data_items_content_hash ON data_items(content_hash);
        CREATE INDEX IF NOT EXISTS idx_data_items_quality_score ON data_items(quality_score);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(create_tables_sql)
            self.logger.info("Database tables created")
    
    async def register_dataset(
        self,
        name: str,
        description: str,
        data_type: str,
        metadata: Dict[str, Any] = None
    ) -> int:
        """
        注册新数据集
        
        Args:
            name: 数据集名称
            description: 描述
            data_type: 数据类型
            metadata: 元数据
            
        Returns:
            int: 数据集ID
        """
        async with self.pool.acquire() as conn:
            dataset_id = await conn.fetchval(
                """
                INSERT INTO datasets (name, description, data_type, metadata)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (name) DO UPDATE 
                SET description = EXCLUDED.description,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING dataset_id
                """,
                name, description, data_type, json.dumps(metadata or {})
            )
            
            self.logger.info(f"Registered dataset: {name} (ID: {dataset_id})")
            return dataset_id
    
    async def get_dataset(self, dataset_id: int) -> Optional[DatasetInfo]:
        """
        获取数据集信息
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            Optional[DatasetInfo]: 数据集信息
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM datasets WHERE dataset_id = $1
                """,
                dataset_id
            )
            
            if row:
                return DatasetInfo(
                    dataset_id=row['dataset_id'],
                    name=row['name'],
                    description=row['description'],
                    data_type=row['data_type'],
                    total_items=row['total_items'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    metadata=row['metadata']
                )
            
            return None
    
    async def insert_item(self, item: Dict[str, Any]) -> bool:
        """
        插入单条数据项元数据
        
        Args:
            item: 数据项字典
            
        Returns:
            bool: 是否成功
        """
        async with self.pool.acquire() as conn:
            try:
                await conn.execute(
                    """
                    INSERT INTO data_items 
                    (item_id, dataset_id, data_type, object_path, content_hash, 
                     status, quality_score, labels, metadata, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (item_id) DO UPDATE 
                    SET status = EXCLUDED.status,
                        quality_score = EXCLUDED.quality_score,
                        labels = EXCLUDED.labels,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    item['data_id'],
                    item.get('dataset_id'),
                    item['data_type'],
                    item['object_path'],
                    item.get('content_hash'),
                    item.get('status', 'raw'),
                    item.get('quality_score'),
                    json.dumps(item.get('labels', {})),
                    json.dumps(item.get('metadata', {})),
                    json.dumps(item.get('source', {}))
                )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to insert item {item['data_id']}: {e}")
                return False
    
    async def insert_items(self, items: List[Dict[str, Any]]) -> int:
        """
        批量插入数据项元数据
        
        Args:
            items: 数据项列表
            
        Returns:
            int: 成功插入的数量
        """
        success_count = 0
        
        async with self.pool.acquire() as conn:
            # 使用事务批量插入
            async with conn.transaction():
                for item in items:
                    try:
                        await conn.execute(
                            """
                            INSERT INTO data_items 
                            (item_id, dataset_id, data_type, object_path, content_hash, 
                             status, quality_score, labels, metadata, source)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            ON CONFLICT (item_id) DO UPDATE 
                            SET status = EXCLUDED.status,
                                quality_score = EXCLUDED.quality_score,
                                labels = EXCLUDED.labels,
                                metadata = EXCLUDED.metadata,
                                updated_at = CURRENT_TIMESTAMP
                            """,
                            item['data_id'],
                            item.get('dataset_id'),
                            item['data_type'],
                            item['object_path'],
                            item.get('content_hash'),
                            item.get('status', 'raw'),
                            item.get('quality_score'),
                            json.dumps(item.get('labels', {})),
                            json.dumps(item.get('metadata', {})),
                            json.dumps(item.get('source', {}))
                        )
                        success_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to insert item {item['data_id']}: {e}")
        
        return success_count
    
    async def update_item_status(
        self,
        item_id: str,
        status: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        更新数据项的处理状态和元数据
        
        Args:
            item_id: 数据项ID
            status: 新状态
            metadata: 额外元数据
            
        Returns:
            bool: 是否成功
        """
        async with self.pool.acquire() as conn:
            try:
                if metadata:
                    await conn.execute(
                        """
                        UPDATE data_items 
                        SET status = $1, 
                            metadata = metadata || $2,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE item_id = $3
                        """,
                        status, json.dumps(metadata), item_id
                    )
                else:
                    await conn.execute(
                        """
                        UPDATE data_items 
                        SET status = $1, updated_at = CURRENT_TIMESTAMP
                        WHERE item_id = $2
                        """,
                        status, item_id
                    )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to update item {item_id}: {e}")
                return False
    
    async def query_items(
        self,
        filters: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[DataItemMeta]:
        """
        按条件查询数据项
        
        Args:
            filters: 过滤条件
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            List[DataItemMeta]: 数据项列表
        """
        # 构建查询条件
        conditions = []
        params = []
        param_idx = 1
        
        if 'dataset_id' in filters:
            conditions.append(f"dataset_id = ${param_idx}")
            params.append(filters['dataset_id'])
            param_idx += 1
        
        if 'status' in filters:
            conditions.append(f"status = ${param_idx}")
            params.append(filters['status'])
            param_idx += 1
        
        if 'data_type' in filters:
            conditions.append(f"data_type = ${param_idx}")
            params.append(filters['data_type'])
            param_idx += 1
        
        if 'min_quality_score' in filters:
            conditions.append(f"quality_score >= ${param_idx}")
            params.append(filters['min_quality_score'])
            param_idx += 1
        
        if 'max_quality_score' in filters:
            conditions.append(f"quality_score <= ${param_idx}")
            params.append(filters['max_quality_score'])
            param_idx += 1
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT * FROM data_items 
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        
        params.extend([limit, offset])
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            items = []
            for row in rows:
                items.append(DataItemMeta(
                    item_id=row['item_id'],
                    dataset_id=row['dataset_id'],
                    data_type=row['data_type'],
                    object_path=row['object_path'],
                    status=row['status'],
                    quality_score=row['quality_score'],
                    labels=row['labels'],
                    metadata=row['metadata'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                ))
            
            return items
    
    async def get_dataset_statistics(self, dataset_id: int) -> DatasetStats:
        """
        获取数据集统计信息
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            DatasetStats: 统计信息
        """
        async with self.pool.acquire() as conn:
            # 总数
            total_items = await conn.fetchval(
                "SELECT COUNT(*) FROM data_items WHERE dataset_id = $1",
                dataset_id
            )
            
            # 状态分布
            status_rows = await conn.fetch(
                """
                SELECT status, COUNT(*) as count 
                FROM data_items 
                WHERE dataset_id = $1 
                GROUP BY status
                """,
                dataset_id
            )
            status_distribution = {row['status']: row['count'] for row in status_rows}
            
            # 数据类型分布
            type_rows = await conn.fetch(
                """
                SELECT data_type, COUNT(*) as count 
                FROM data_items 
                WHERE dataset_id = $1 
                GROUP BY data_type
                """,
                dataset_id
            )
            data_type_distribution = {row['data_type']: row['count'] for row in type_rows}
            
            # 平均质量分数
            avg_quality = await conn.fetchval(
                """
                SELECT AVG(quality_score) 
                FROM data_items 
                WHERE dataset_id = $1 AND quality_score IS NOT NULL
                """,
                dataset_id
            )
            
            # 质量分布
            quality_rows = await conn.fetch(
                """
                SELECT 
                    CASE 
                        WHEN quality_score < 0.2 THEN '0.0-0.2'
                        WHEN quality_score < 0.4 THEN '0.2-0.4'
                        WHEN quality_score < 0.6 THEN '0.4-0.6'
                        WHEN quality_score < 0.8 THEN '0.6-0.8'
                        ELSE '0.8-1.0'
                    END as quality_range,
                    COUNT(*) as count
                FROM data_items
                WHERE dataset_id = $1 AND quality_score IS NOT NULL
                GROUP BY quality_range
                """,
                dataset_id
            )
            quality_distribution = {row['quality_range']: row['count'] for row in quality_rows}
            
            return DatasetStats(
                total_items=total_items or 0,
                status_distribution=status_distribution,
                quality_distribution=quality_distribution,
                data_type_distribution=data_type_distribution,
                avg_quality_score=avg_quality or 0.0
            )
    
    async def delete_dataset(self, dataset_id: int) -> bool:
        """
        删除数据集及其所有数据项
        
        Args:
            dataset_id: 数据集ID
            
        Returns:
            bool: 是否成功
        """
        async with self.pool.acquire() as conn:
            try:
                # 删除数据项
                await conn.execute(
                    "DELETE FROM data_items WHERE dataset_id = $1",
                    dataset_id
                )
                
                # 删除数据集
                await conn.execute(
                    "DELETE FROM datasets WHERE dataset_id = $1",
                    dataset_id
                )
                
                self.logger.info(f"Deleted dataset: {dataset_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to delete dataset {dataset_id}: {e}")
                return False