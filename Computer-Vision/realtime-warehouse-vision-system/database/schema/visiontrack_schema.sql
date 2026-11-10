-- =====================================================================
-- VisionTrack CURRENT DATABASE SCHEMA (schema only — no seed data)
-- Database: visiontrack_local
--
-- This file represents the current normalized structure used by the
-- current dashboard and seed files:
--   orders -> order_items -> order_item_products -> order_boxes -> boxes
--   order_lifecycle_events = central operational source of truth
--
-- IMPORTANT:
-- Running the complete file DROPS and recreates the listed objects.
-- Take a backup before running it against a database containing data.
-- =====================================================================

USE `visiontrack_local`;

SET NAMES utf8mb4;
SET SQL_SAFE_UPDATES = 0;
SET FOREIGN_KEY_CHECKS = 0;

-- ---------------------------------------------------------------------
-- Drop current views
-- ---------------------------------------------------------------------
DROP VIEW IF EXISTS `v_latest_video_events`;
DROP VIEW IF EXISTS `v_latest_video_boxes`;
DROP VIEW IF EXISTS `v_latest_video_pallets`;
DROP VIEW IF EXISTS `v_latest_video_run`;
DROP VIEW IF EXISTS `vw_pallet_trailer_routes`;
DROP VIEW IF EXISTS `vw_conveyor_boxes`;
DROP VIEW IF EXISTS `vw_box_current_lifecycle`;
DROP VIEW IF EXISTS `vw_box_lifecycle_history`;
DROP VIEW IF EXISTS `vw_order_current_lifecycle`;
DROP VIEW IF EXISTS `vw_order_lifecycle_history`;
DROP VIEW IF EXISTS `v_box_current_location`;
DROP VIEW IF EXISTS `v_box_journey`;
DROP VIEW IF EXISTS `v_pallet_box_count`;
DROP VIEW IF EXISTS `v_order_details`;

-- ---------------------------------------------------------------------
-- Drop triggers
-- ---------------------------------------------------------------------
DROP TRIGGER IF EXISTS `trg_boxes_after_update_lifecycle`;
DROP TRIGGER IF EXISTS `trg_order_boxes_after_insert_lifecycle`;
DROP TRIGGER IF EXISTS `trg_orders_after_update_lifecycle`;
DROP TRIGGER IF EXISTS `trg_orders_after_insert_lifecycle`;

-- ---------------------------------------------------------------------
-- Drop tables: children before parents
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS `video_processing_metrics`;
DROP TABLE IF EXISTS `video_final_pallets`;
DROP TABLE IF EXISTS `video_final_boxes`;
DROP TABLE IF EXISTS `video_events`;
DROP TABLE IF EXISTS `video_processing_runs`;
DROP TABLE IF EXISTS `order_lifecycle_events`;
DROP TABLE IF EXISTS `audit_logs`;
DROP TABLE IF EXISTS `events`;
DROP TABLE IF EXISTS `order_boxes`;
DROP TABLE IF EXISTS `order_item_products`;
DROP TABLE IF EXISTS `order_items`;
DROP TABLE IF EXISTS `trailer_truck_assignments`;
DROP TABLE IF EXISTS `trailer_dock_assignments`;
DROP TABLE IF EXISTS `boxes`;
DROP TABLE IF EXISTS `orders`;
DROP TABLE IF EXISTS `docks`;
DROP TABLE IF EXISTS `trucks`;
DROP TABLE IF EXISTS `trailers`;
DROP TABLE IF EXISTS `conveyors`;
DROP TABLE IF EXISTS `pallets`;
DROP TABLE IF EXISTS `products`;
DROP TABLE IF EXISTS `customers`;
DROP TABLE IF EXISTS `zones`;
DROP TABLE IF EXISTS `team_users`;

SET FOREIGN_KEY_CHECKS = 1;

-- =====================================================================
-- 1. MASTER TABLES
-- =====================================================================

CREATE TABLE `team_users` (
    `user_id` INT AUTO_INCREMENT PRIMARY KEY,
    `username` VARCHAR(50) NOT NULL,
    `full_name` VARCHAR(100) NOT NULL,
    `email` VARCHAR(150) NULL,
    `role` ENUM('ADMIN','DEVELOPER','VIEWER') NOT NULL DEFAULT 'DEVELOPER',
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT `uq_team_users_username` UNIQUE (`username`),
    INDEX `idx_team_users_role` (`role`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `zones` (
    `zone_id` INT AUTO_INCREMENT PRIMARY KEY,
    `zone_code` VARCHAR(50) NOT NULL,
    `zone_name` VARCHAR(100) NOT NULL,
    `description` TEXT NULL,
    `zone_type` VARCHAR(40) NOT NULL DEFAULT 'FLOOR',
    `polygon_points` TEXT NULL,
    `is_active` TINYINT(1) NOT NULL DEFAULT 1,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT `uq_zones_zone_code` UNIQUE (`zone_code`),
    INDEX `idx_zones_zone_name` (`zone_name`),
    INDEX `idx_zones_zone_type` (`zone_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `customers` (
    `customer_id` INT AUTO_INCREMENT PRIMARY KEY,
    `customer_code` VARCHAR(50) NOT NULL,
    `customer_name` VARCHAR(150) NOT NULL,
    `email` VARCHAR(150) NULL,
    `phone` VARCHAR(20) NULL,
    `address` TEXT NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT `uq_customers_customer_code` UNIQUE (`customer_code`),
    INDEX `idx_customers_customer_name` (`customer_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `products` (
    `product_id` INT AUTO_INCREMENT PRIMARY KEY,
    `product_code` VARCHAR(50) NOT NULL,
    `product_name` VARCHAR(150) NOT NULL,
    `length_cm` DECIMAL(10,2) NOT NULL,
    `breadth_cm` DECIMAL(10,2) NOT NULL,
    `height_cm` DECIMAL(10,2) NOT NULL,
    `weight_kg` DECIMAL(10,2) NULL,
    `volume_cm3` DECIMAL(12,2)
        GENERATED ALWAYS AS (`length_cm` * `breadth_cm` * `height_cm`) STORED,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT `uq_products_product_code` UNIQUE (`product_code`),
    INDEX `idx_products_product_name` (`product_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `pallets` (
    `pallet_id` INT AUTO_INCREMENT PRIMARY KEY,
    `pallet_code` VARCHAR(50) NOT NULL,
    `pallet_status` ENUM('EMPTY','IN_USE','FULL','MOVED','DISPATCHED')
        NULL DEFAULT 'EMPTY',
    `max_weight_kg` DECIMAL(10,2) NULL DEFAULT 1000.00,
    `max_volume_cm3` DECIMAL(12,2) NULL,
    `max_height_cm` DECIMAL(10,2) NULL,
    `created_by` INT NULL,
    `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT `uq_pallets_pallet_code` UNIQUE (`pallet_code`),
    CONSTRAINT `fk_pallets_created_by`
        FOREIGN KEY (`created_by`) REFERENCES `team_users` (`user_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    INDEX `idx_pallets_status` (`pallet_status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `conveyors` (
    `conveyor_id` INT AUTO_INCREMENT PRIMARY KEY,
    `conveyor_code` VARCHAR(50) NOT NULL,
    `conveyor_name` VARCHAR(100) NULL,
    `zone_id` INT NULL,
    `status` VARCHAR(30) NOT NULL DEFAULT 'ACTIVE',
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT `uq_conveyors_conveyor_code` UNIQUE (`conveyor_code`),
    CONSTRAINT `fk_conveyors_zone`
        FOREIGN KEY (`zone_id`) REFERENCES `zones` (`zone_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    INDEX `idx_conveyors_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `trailers` (
    `trailer_id` INT AUTO_INCREMENT PRIMARY KEY,
    `trailer_code` VARCHAR(50) NOT NULL,
    `registration_number` VARCHAR(50) NULL,
    `carrier_name` VARCHAR(120) NULL,
    `trailer_type` ENUM(
        'DRY_VAN','REFRIGERATED','FLATBED','CONTAINER','OTHER'
    ) NOT NULL DEFAULT 'DRY_VAN',
    `capacity_pallets` INT UNSIGNED NULL,
    `trailer_status` ENUM(
        'EXPECTED','WAITING','AT_DOCK','LOADING','LOADED',
        'DEPARTED','MAINTENANCE'
    ) NOT NULL DEFAULT 'EXPECTED',
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT `uq_trailers_trailer_code` UNIQUE (`trailer_code`),
    CONSTRAINT `uq_trailers_registration_number` UNIQUE (`registration_number`),
    INDEX `idx_trailers_status` (`trailer_status`),
    INDEX `idx_trailers_carrier` (`carrier_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `trucks` (
    `truck_id` INT AUTO_INCREMENT PRIMARY KEY,
    `truck_number` VARCHAR(50) NOT NULL,
    `carrier_name` VARCHAR(100) NULL,
    `status` ENUM(
        'EXPECTED','WAITING','AT_DOCK','LOADING','UNLOADING','DISPATCHED'
    ) NOT NULL DEFAULT 'EXPECTED',
    `arrival_time` DATETIME NULL,
    `dock_time` DATETIME NULL,
    `departure_time` DATETIME NULL,
    `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT `uq_trucks_truck_number` UNIQUE (`truck_number`),
    INDEX `idx_trucks_status` (`status`),
    INDEX `idx_trucks_arrival_time` (`arrival_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `docks` (
    `dock_id` INT AUTO_INCREMENT PRIMARY KEY,
    `dock_code` VARCHAR(30) NOT NULL,
    `dock_name` VARCHAR(100) NOT NULL,
    `zone_id` INT NOT NULL,
    `dock_status` VARCHAR(40) NOT NULL DEFAULT 'AVAILABLE',
    `max_trailers` INT NOT NULL DEFAULT 1,
    `is_active` TINYINT(1) NOT NULL DEFAULT 1,
    `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT `uq_docks_dock_code` UNIQUE (`dock_code`),
    CONSTRAINT `uq_docks_zone` UNIQUE (`zone_id`),
    CONSTRAINT `fk_docks_zone`
        FOREIGN KEY (`zone_id`) REFERENCES `zones` (`zone_id`)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    INDEX `idx_docks_status` (`dock_status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================================
-- 2. ORDER AND PACKING TABLES
-- =====================================================================

CREATE TABLE `orders` (
    `order_id` INT AUTO_INCREMENT PRIMARY KEY,
    `order_code` VARCHAR(50) NOT NULL,
    `customer_id` INT NOT NULL,
    `order_status` ENUM('CREATED','PACKING','READY','DISPATCHED','CANCELLED')
        NOT NULL DEFAULT 'CREATED',
    `order_length_cm` DECIMAL(10,2) NULL,
    `order_breadth_cm` DECIMAL(10,2) NULL,
    `order_height_cm` DECIMAL(10,2) NULL,
    `order_weight_kg` DECIMAL(10,2) NULL,
    `created_by` INT NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT `uq_orders_order_code` UNIQUE (`order_code`),
    CONSTRAINT `fk_orders_customer`
        FOREIGN KEY (`customer_id`) REFERENCES `customers` (`customer_id`)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT `fk_orders_created_by`
        FOREIGN KEY (`created_by`) REFERENCES `team_users` (`user_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    INDEX `idx_orders_customer` (`customer_id`),
    INDEX `idx_orders_status` (`order_status`),
    INDEX `idx_orders_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `order_items` (
    `order_item_id` INT AUTO_INCREMENT PRIMARY KEY,
    `order_id` INT NOT NULL,
    `item_code` VARCHAR(50) NULL,
    `item_description` VARCHAR(255) NULL,
    `item_status` VARCHAR(30) NOT NULL DEFAULT 'CREATED',
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT `fk_order_items_order`
        FOREIGN KEY (`order_id`) REFERENCES `orders` (`order_id`)
        ON UPDATE CASCADE ON DELETE CASCADE,
    INDEX `idx_order_items_order` (`order_id`),
    INDEX `idx_order_items_status` (`item_status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `order_item_products` (
    `order_item_product_id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `order_item_id` INT NOT NULL,
    `product_id` INT NOT NULL,
    `ordered_quantity` INT NOT NULL DEFAULT 1,
    `fulfilled_quantity` INT NOT NULL DEFAULT 0,
    `product_status` VARCHAR(30) NOT NULL DEFAULT 'ORDERED',
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT `fk_oip_order_item`
        FOREIGN KEY (`order_item_id`) REFERENCES `order_items` (`order_item_id`)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT `fk_oip_product`
        FOREIGN KEY (`product_id`) REFERENCES `products` (`product_id`)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT `uq_oip_item_product` UNIQUE (`order_item_id`, `product_id`),
    CONSTRAINT `chk_oip_ordered_quantity` CHECK (`ordered_quantity` >= 0),
    CONSTRAINT `chk_oip_fulfilled_quantity` CHECK (`fulfilled_quantity` >= 0),
    INDEX `idx_oip_order_item` (`order_item_id`),
    INDEX `idx_oip_product` (`product_id`),
    INDEX `idx_oip_status` (`product_status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `boxes` (
    `box_id` INT AUTO_INCREMENT PRIMARY KEY,
    `box_code` VARCHAR(50) NOT NULL,
    `length_cm` DECIMAL(10,2) NOT NULL,
    `breadth_cm` DECIMAL(10,2) NOT NULL,
    `height_cm` DECIMAL(10,2) NOT NULL,
    `weight_kg` DECIMAL(10,2) NULL,
    `volume_cm3` DECIMAL(12,2)
        GENERATED ALWAYS AS (`length_cm` * `breadth_cm` * `height_cm`) STORED,
    `box_status` ENUM(
        'RECEIVED','AT_DOCKING_STATION','ON_CONVEYOR','ON_FLOOR',
        'ON_PALLET','IN_STOCKING_AREA','MOVED','REMOVED','MISSING',
        'DISPATCHED'
    ) NOT NULL DEFAULT 'RECEIVED',
    `created_by` INT NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT `uq_boxes_box_code` UNIQUE (`box_code`),
    CONSTRAINT `fk_boxes_created_by`
        FOREIGN KEY (`created_by`) REFERENCES `team_users` (`user_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    INDEX `idx_boxes_status` (`box_status`),
    INDEX `idx_boxes_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `order_boxes` (
    `order_box_id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `box_id` INT NOT NULL,
    `order_item_product_id` BIGINT NOT NULL,
    `packed_quantity` INT NOT NULL DEFAULT 1,
    `packed_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `packing_status` VARCHAR(30) NOT NULL DEFAULT 'PACKED',
    `remarks` VARCHAR(500) NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT `fk_order_boxes_box`
        FOREIGN KEY (`box_id`) REFERENCES `boxes` (`box_id`)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT `fk_order_boxes_oip`
        FOREIGN KEY (`order_item_product_id`)
        REFERENCES `order_item_products` (`order_item_product_id`)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT `uq_order_boxes_box_oip`
        UNIQUE (`box_id`, `order_item_product_id`),
    CONSTRAINT `chk_order_boxes_quantity` CHECK (`packed_quantity` >= 0),
    INDEX `idx_order_boxes_box` (`box_id`),
    INDEX `idx_order_boxes_oip` (`order_item_product_id`),
    INDEX `idx_order_boxes_packed_at` (`packed_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================================
-- 3. TRAILER, DOCK AND TRUCK ASSOCIATIONS
-- =====================================================================

CREATE TABLE `trailer_dock_assignments` (
    `trailer_dock_assignment_id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `trailer_id` INT NOT NULL,
    `dock_id` INT NOT NULL,
    `operation_type` ENUM('LOADING','UNLOADING','BOTH')
        NOT NULL DEFAULT 'LOADING',
    `scheduled_from` DATETIME NOT NULL,
    `scheduled_to` DATETIME NOT NULL,
    `actual_start` DATETIME NULL,
    `actual_end` DATETIME NULL,
    `assignment_status` ENUM('SCHEDULED','ACTIVE','COMPLETED','CANCELLED')
        NOT NULL DEFAULT 'SCHEDULED',
    `notes` VARCHAR(500) NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT `fk_tda_trailer`
        FOREIGN KEY (`trailer_id`) REFERENCES `trailers` (`trailer_id`)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT `fk_tda_dock`
        FOREIGN KEY (`dock_id`) REFERENCES `docks` (`dock_id`)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT `chk_tda_schedule` CHECK (`scheduled_to` > `scheduled_from`),
    INDEX `idx_tda_trailer` (`trailer_id`),
    INDEX `idx_tda_dock` (`dock_id`),
    INDEX `idx_tda_status` (`assignment_status`),
    INDEX `idx_tda_schedule` (`scheduled_from`, `scheduled_to`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `trailer_truck_assignments` (
    `trailer_truck_assignment_id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `trailer_id` INT NOT NULL,
    `truck_id` INT NOT NULL,
    `assignment_status` VARCHAR(30) NOT NULL DEFAULT 'ASSIGNED',
    `assigned_at` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `attached_at` DATETIME NULL,
    `detached_at` DATETIME NULL,
    `remarks` VARCHAR(500) NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    CONSTRAINT `fk_tta_trailer`
        FOREIGN KEY (`trailer_id`) REFERENCES `trailers` (`trailer_id`)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT `fk_tta_truck`
        FOREIGN KEY (`truck_id`) REFERENCES `trucks` (`truck_id`)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    INDEX `idx_tta_trailer` (`trailer_id`),
    INDEX `idx_tta_truck` (`truck_id`),
    INDEX `idx_tta_assigned_at` (`assigned_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================================
-- 4. GENERIC MOVEMENT AND AUDIT TABLES
-- =====================================================================

CREATE TABLE `events` (
    `event_id` INT AUTO_INCREMENT PRIMARY KEY,
    `event_type` ENUM(
        'ADD','MOVE','SWAP','STACK','REMOVE','RE_ADD','ZONE_TRANSFER',
        'CONVEYOR_ENTRY','CONVEYOR_EXIT','DISPATCH'
    ) NOT NULL,
    `box_id` INT NOT NULL,
    `from_pallet_id` INT NULL,
    `to_pallet_id` INT NULL,
    `from_zone_id` INT NULL,
    `to_zone_id` INT NULL,
    `event_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `remarks` TEXT NULL,

    CONSTRAINT `fk_events_box`
        FOREIGN KEY (`box_id`) REFERENCES `boxes` (`box_id`)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT `fk_events_from_pallet`
        FOREIGN KEY (`from_pallet_id`) REFERENCES `pallets` (`pallet_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT `fk_events_to_pallet`
        FOREIGN KEY (`to_pallet_id`) REFERENCES `pallets` (`pallet_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT `fk_events_from_zone`
        FOREIGN KEY (`from_zone_id`) REFERENCES `zones` (`zone_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT `fk_events_to_zone`
        FOREIGN KEY (`to_zone_id`) REFERENCES `zones` (`zone_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    INDEX `idx_events_box_time` (`box_id`, `event_time`),
    INDEX `idx_events_type` (`event_type`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `audit_logs` (
    `audit_id` INT AUTO_INCREMENT PRIMARY KEY,
    `table_name` VARCHAR(100) NULL,
    `record_id` INT NULL,
    `action_type` ENUM('INSERT','UPDATE','DELETE') NULL,
    `changed_by` INT NULL,
    `change_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `old_value` TEXT NULL,
    `new_value` TEXT NULL,

    CONSTRAINT `fk_audit_logs_changed_by`
        FOREIGN KEY (`changed_by`) REFERENCES `team_users` (`user_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    INDEX `idx_audit_table_record` (`table_name`, `record_id`),
    INDEX `idx_audit_change_time` (`change_time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================================
-- 5. CENTRAL ORDER / BOX LIFECYCLE TABLE
-- =====================================================================

CREATE TABLE `order_lifecycle_events` (
    `order_lifecycle_event_id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `entity_type` VARCHAR(30) NOT NULL DEFAULT 'ORDER',
    `lifecycle_status` VARCHAR(50) NOT NULL,
    `event_type` VARCHAR(50) NOT NULL DEFAULT 'STATUS_UPDATE',
    `order_id` INT NULL,
    `order_item_product_id` BIGINT NULL,
    `box_id` INT NULL,
    `conveyor_id` INT NULL,
    `pallet_id` INT NULL,
    `zone_id` INT NULL,
    `dock_id` INT NULL,
    `trailer_id` INT NULL,
    `truck_id` INT NULL,
    `trailer_dock_assignment_id` BIGINT NULL,
    `trailer_truck_assignment_id` BIGINT NULL,
    `event_source` VARCHAR(30) NOT NULL DEFAULT 'SYSTEM',
    `source_reference` VARCHAR(150) NULL,
    `confidence_score` DECIMAL(5,2) NULL,
    `qr_read_success` TINYINT(1) NULL,
    `severity` VARCHAR(20) NOT NULL DEFAULT 'NORMAL',
    `event_time` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `recorded_by` INT NULL,
    `remarks` VARCHAR(500) NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT `fk_ole_order`
        FOREIGN KEY (`order_id`) REFERENCES `orders` (`order_id`)
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT `fk_ole_oip`
        FOREIGN KEY (`order_item_product_id`)
        REFERENCES `order_item_products` (`order_item_product_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT `fk_ole_box`
        FOREIGN KEY (`box_id`) REFERENCES `boxes` (`box_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT `fk_ole_conveyor`
        FOREIGN KEY (`conveyor_id`) REFERENCES `conveyors` (`conveyor_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT `fk_ole_pallet`
        FOREIGN KEY (`pallet_id`) REFERENCES `pallets` (`pallet_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT `fk_ole_zone`
        FOREIGN KEY (`zone_id`) REFERENCES `zones` (`zone_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT `fk_ole_dock`
        FOREIGN KEY (`dock_id`) REFERENCES `docks` (`dock_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT `fk_ole_trailer`
        FOREIGN KEY (`trailer_id`) REFERENCES `trailers` (`trailer_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT `fk_ole_truck`
        FOREIGN KEY (`truck_id`) REFERENCES `trucks` (`truck_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT `fk_ole_tda`
        FOREIGN KEY (`trailer_dock_assignment_id`)
        REFERENCES `trailer_dock_assignments` (`trailer_dock_assignment_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT `fk_ole_tta`
        FOREIGN KEY (`trailer_truck_assignment_id`)
        REFERENCES `trailer_truck_assignments` (`trailer_truck_assignment_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,
    CONSTRAINT `fk_ole_recorded_by`
        FOREIGN KEY (`recorded_by`) REFERENCES `team_users` (`user_id`)
        ON UPDATE CASCADE ON DELETE SET NULL,

    INDEX `idx_ole_entity_type` (`entity_type`),
    INDEX `idx_ole_status` (`lifecycle_status`),
    INDEX `idx_ole_order_time` (`order_id`, `event_time`),
    INDEX `idx_ole_box_time` (`box_id`, `event_time`),
    INDEX `idx_ole_oip` (`order_item_product_id`),
    INDEX `idx_ole_conveyor` (`conveyor_id`),
    INDEX `idx_ole_pallet` (`pallet_id`),
    INDEX `idx_ole_zone` (`zone_id`),
    INDEX `idx_ole_dock` (`dock_id`),
    INDEX `idx_ole_trailer` (`trailer_id`),
    INDEX `idx_ole_truck` (`truck_id`),
    INDEX `idx_ole_source_reference` (`source_reference`),
    INDEX `idx_ole_severity` (`severity`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================================
-- 6. COMPUTER-VISION TABLES
-- =====================================================================

CREATE TABLE `video_processing_runs` (
    `run_id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `video_name` VARCHAR(255) NOT NULL,
    `input_video_path` TEXT NULL,
    `model_path` TEXT NULL,
    `device` VARCHAR(100) NULL,
    `gpu_name` VARCHAR(255) NULL,
    `source_width` INT NULL,
    `source_height` INT NULL,
    `source_fps` DECIMAL(10,3) NULL,
    `video_duration_sec` DECIMAL(12,3) NULL,
    `frame_skip` INT NULL,
    `process_width` INT NULL,
    `process_height` INT NULL,
    `yolo_imgsz` INT NULL,
    `processed_frames` INT NULL,
    `source_frames_covered` INT NULL,
    `total_processing_time_sec` DECIMAL(12,3) NULL,
    `end_to_end_time_sec` DECIMAL(12,3) NULL,
    `speed_vs_realtime` DECIMAL(12,3) NULL,
    `realtime_factor` DECIMAL(12,3) NULL,
    `status` ENUM('RUNNING','COMPLETED','FAILED') DEFAULT 'COMPLETED',
    `started_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `completed_at` TIMESTAMP NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX `idx_video_runs_status` (`status`),
    INDEX `idx_video_runs_created_at` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `video_events` (
    `video_event_id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `run_id` BIGINT NOT NULL,
    `processed_index` INT NULL,
    `video_time` VARCHAR(20) NULL,
    `timestamp_sec` DECIMAL(12,3) NULL,
    `event_type` VARCHAR(80) NULL,
    `severity` VARCHAR(30) NULL,
    `box_code` VARCHAR(50) NULL,
    `pallet_code` VARCHAR(50) NULL,
    `from_pallet_code` VARCHAR(50) NULL,
    `to_pallet_code` VARCHAR(50) NULL,
    `description` TEXT NULL,
    `event_created_at` DATETIME NULL,
    `inserted_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT `fk_video_events_run`
        FOREIGN KEY (`run_id`) REFERENCES `video_processing_runs` (`run_id`)
        ON UPDATE CASCADE ON DELETE CASCADE,
    INDEX `idx_video_events_run_type` (`run_id`, `event_type`),
    INDEX `idx_video_events_box` (`box_code`),
    INDEX `idx_video_events_pallet`
        (`pallet_code`, `from_pallet_code`, `to_pallet_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `video_final_boxes` (
    `final_box_id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `run_id` BIGINT NOT NULL,
    `box_code` VARCHAR(50) NOT NULL,
    `status` VARCHAR(80) NULL,
    `pallet_code` VARCHAR(50) NULL,
    `slot_no` INT NULL,
    `row_no` INT NULL,
    `column_no` INT NULL,
    `layer_no` INT NULL,
    `length_cm` DECIMAL(10,2) NULL,
    `width_cm` DECIMAL(10,2) NULL,
    `height_cm` DECIMAL(10,2) NULL,
    `weight_kg` DECIMAL(10,2) NULL,
    `volume_cm3` DECIMAL(12,2) NULL,
    `inserted_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT `fk_video_final_boxes_run`
        FOREIGN KEY (`run_id`) REFERENCES `video_processing_runs` (`run_id`)
        ON UPDATE CASCADE ON DELETE CASCADE,
    INDEX `idx_video_final_boxes_run` (`run_id`, `pallet_code`, `status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `video_final_pallets` (
    `final_pallet_id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `run_id` BIGINT NOT NULL,
    `pallet_code` VARCHAR(50) NOT NULL,
    `active` TINYINT(1) NULL DEFAULT 0,
    `status` VARCHAR(80) NULL,
    `box_count` INT NULL,
    `max_boxes` INT NULL,
    `total_weight_kg` DECIMAL(12,3) NULL,
    `max_weight_kg` DECIMAL(12,3) NULL,
    `weight_utilization_percent` DECIMAL(8,2) NULL,
    `total_volume_cm3` DECIMAL(14,3) NULL,
    `max_volume_cm3` DECIMAL(14,3) NULL,
    `volume_efficiency_percent` DECIMAL(8,2) NULL,
    `max_stack_height_cm` DECIMAL(10,3) NULL,
    `max_height_cm` DECIMAL(10,3) NULL,
    `height_utilization_percent` DECIMAL(8,2) NULL,
    `bottleneck_utilization_percent` DECIMAL(8,2) NULL,
    `reasons` TEXT NULL,
    `inserted_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT `fk_video_final_pallets_run`
        FOREIGN KEY (`run_id`) REFERENCES `video_processing_runs` (`run_id`)
        ON UPDATE CASCADE ON DELETE CASCADE,
    INDEX `idx_video_final_pallets_run` (`run_id`, `pallet_code`, `status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE `video_processing_metrics` (
    `metric_id` BIGINT AUTO_INCREMENT PRIMARY KEY,
    `run_id` BIGINT NOT NULL,
    `metric_name` VARCHAR(128) NOT NULL,
    `metric_value` TEXT NULL,
    `inserted_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT `fk_video_metrics_run`
        FOREIGN KEY (`run_id`) REFERENCES `video_processing_runs` (`run_id`)
        ON UPDATE CASCADE ON DELETE CASCADE,
    INDEX `idx_video_metrics_run` (`run_id`, `metric_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================================
-- 7. CURRENT-SCHEMA TRIGGERS
-- =====================================================================

DELIMITER $$

CREATE TRIGGER `trg_orders_after_insert_lifecycle`
AFTER INSERT ON `orders`
FOR EACH ROW
BEGIN
    INSERT INTO `order_lifecycle_events` (
        `entity_type`, `lifecycle_status`, `event_type`, `order_id`,
        `event_source`, `source_reference`, `severity`, `event_time`,
        `recorded_by`, `remarks`
    ) VALUES (
        'ORDER', NEW.`order_status`, 'ORDER_CREATED', NEW.`order_id`,
        'SYSTEM', CONCAT('orders-insert-', NEW.`order_id`), 'NORMAL', NOW(),
        NEW.`created_by`, CONCAT('Order created with status ', NEW.`order_status`)
    );
END$$

CREATE TRIGGER `trg_orders_after_update_lifecycle`
AFTER UPDATE ON `orders`
FOR EACH ROW
BEGIN
    IF NOT (OLD.`order_status` <=> NEW.`order_status`) THEN
        INSERT INTO `order_lifecycle_events` (
            `entity_type`, `lifecycle_status`, `event_type`, `order_id`,
            `event_source`, `source_reference`, `severity`, `event_time`,
            `recorded_by`, `remarks`
        ) VALUES (
            'ORDER', NEW.`order_status`, 'ORDER_STATUS_CHANGED', NEW.`order_id`,
            'SYSTEM', CONCAT('orders-update-', NEW.`order_id`, '-', UNIX_TIMESTAMP()),
            CASE WHEN NEW.`order_status` = 'CANCELLED' THEN 'WARNING' ELSE 'NORMAL' END,
            NOW(), NEW.`created_by`,
            CONCAT('Order status changed from ', OLD.`order_status`, ' to ', NEW.`order_status`)
        );
    END IF;
END$$

CREATE TRIGGER `trg_order_boxes_after_insert_lifecycle`
AFTER INSERT ON `order_boxes`
FOR EACH ROW
BEGIN
    DECLARE v_order_id INT DEFAULT NULL;

    SELECT oi.`order_id`
      INTO v_order_id
      FROM `order_item_products` oip
      JOIN `order_items` oi
        ON oi.`order_item_id` = oip.`order_item_id`
     WHERE oip.`order_item_product_id` = NEW.`order_item_product_id`
     LIMIT 1;

    INSERT INTO `order_lifecycle_events` (
        `entity_type`, `lifecycle_status`, `event_type`, `order_id`,
        `order_item_product_id`, `box_id`, `event_source`,
        `source_reference`, `severity`, `event_time`, `remarks`
    ) VALUES (
        'BOX', 'PACKED', 'PRODUCT_PACKED', v_order_id,
        NEW.`order_item_product_id`, NEW.`box_id`, 'SYSTEM',
        CONCAT('order-box-insert-', NEW.`order_box_id`), 'NORMAL', NEW.`packed_at`,
        CONCAT('Packed quantity: ', NEW.`packed_quantity`)
    );
END$$

CREATE TRIGGER `trg_boxes_after_update_lifecycle`
AFTER UPDATE ON `boxes`
FOR EACH ROW
BEGIN
    IF NOT (OLD.`box_status` <=> NEW.`box_status`) THEN
        INSERT INTO `order_lifecycle_events` (
            `entity_type`, `lifecycle_status`, `event_type`, `box_id`,
            `event_source`, `source_reference`, `severity`, `event_time`,
            `recorded_by`, `remarks`
        ) VALUES (
            'BOX',
            CASE UPPER(NEW.`box_status`)
                WHEN 'RECEIVED' THEN 'RECEIVED'
                WHEN 'AT_DOCKING_STATION' THEN 'AT_DOCK'
                WHEN 'ON_CONVEYOR' THEN 'ON_CONVEYOR'
                WHEN 'ON_PALLET' THEN 'ON_PALLET'
                WHEN 'DISPATCHED' THEN 'DISPATCHED'
                WHEN 'MISSING' THEN 'BOX_EXCEPTION'
                WHEN 'REMOVED' THEN 'BOX_EXCEPTION'
                ELSE 'ON_FLOOR'
            END,
            'BOX_STATUS_CHANGED', NEW.`box_id`, 'SYSTEM',
            CONCAT('boxes-update-', NEW.`box_id`, '-', UNIX_TIMESTAMP()),
            CASE WHEN NEW.`box_status` IN ('MISSING','REMOVED')
                 THEN 'CRITICAL' ELSE 'NORMAL' END,
            NOW(), NEW.`created_by`,
            CONCAT('Box status changed from ', OLD.`box_status`, ' to ', NEW.`box_status`)
        );
    END IF;
END$$

DELIMITER ;

-- =====================================================================
-- 8. DASHBOARD / REPORTING VIEWS
-- =====================================================================

CREATE OR REPLACE VIEW `v_order_details` AS
SELECT
    o.`order_id`,
    o.`order_code`,
    c.`customer_code`,
    c.`customer_name`,
    o.`order_status`,
    COUNT(DISTINCT oi.`order_item_id`) AS `total_order_items`,
    COUNT(DISTINCT oip.`product_id`) AS `total_products`,
    COALESCE(SUM(oip.`ordered_quantity`), 0) AS `ordered_quantity`,
    COALESCE(SUM(oip.`fulfilled_quantity`), 0) AS `fulfilled_quantity`,
    COUNT(DISTINCT ob.`box_id`) AS `total_boxes`,
    o.`created_at`
FROM `orders` o
JOIN `customers` c ON c.`customer_id` = o.`customer_id`
LEFT JOIN `order_items` oi ON oi.`order_id` = o.`order_id`
LEFT JOIN `order_item_products` oip ON oip.`order_item_id` = oi.`order_item_id`
LEFT JOIN `order_boxes` ob
    ON ob.`order_item_product_id` = oip.`order_item_product_id`
GROUP BY
    o.`order_id`, o.`order_code`, c.`customer_code`, c.`customer_name`,
    o.`order_status`, o.`created_at`;

CREATE OR REPLACE VIEW `vw_order_lifecycle_history` AS
SELECT
    ole.*,
    o.`order_code`,
    c.`customer_code`,
    c.`customer_name`
FROM `order_lifecycle_events` ole
LEFT JOIN `orders` o ON o.`order_id` = ole.`order_id`
LEFT JOIN `customers` c ON c.`customer_id` = o.`customer_id`
WHERE ole.`order_id` IS NOT NULL;

CREATE OR REPLACE VIEW `vw_order_current_lifecycle` AS
SELECT history.*
FROM `vw_order_lifecycle_history` history
JOIN (
    SELECT `order_id`, MAX(`order_lifecycle_event_id`) AS `latest_event_id`
    FROM `order_lifecycle_events`
    WHERE `order_id` IS NOT NULL
    GROUP BY `order_id`
) latest
    ON latest.`latest_event_id` = history.`order_lifecycle_event_id`;

CREATE OR REPLACE VIEW `vw_box_lifecycle_history` AS
SELECT
    ole.`order_lifecycle_event_id` AS `box_lifecycle_event_id`,
    ole.`order_id`,
    o.`order_code`,
    ole.`order_item_product_id`,
    pr.`product_code`,
    ole.`box_id`,
    b.`box_code`,
    b.`box_status`,
    ole.`lifecycle_status`,
    ole.`event_type`,
    conv.`conveyor_code`,
    p.`pallet_code`,
    z.`zone_code`,
    z.`zone_name`,
    d.`dock_code`,
    t.`trailer_code`,
    tr.`truck_number`,
    ole.`event_source`,
    ole.`source_reference`,
    ole.`confidence_score`,
    ole.`qr_read_success`,
    ole.`severity`,
    ole.`event_time`,
    ole.`remarks`
FROM `order_lifecycle_events` ole
LEFT JOIN `orders` o ON o.`order_id` = ole.`order_id`
LEFT JOIN `order_item_products` oip
    ON oip.`order_item_product_id` = ole.`order_item_product_id`
LEFT JOIN `products` pr ON pr.`product_id` = oip.`product_id`
LEFT JOIN `boxes` b ON b.`box_id` = ole.`box_id`
LEFT JOIN `conveyors` conv ON conv.`conveyor_id` = ole.`conveyor_id`
LEFT JOIN `pallets` p ON p.`pallet_id` = ole.`pallet_id`
LEFT JOIN `zones` z ON z.`zone_id` = ole.`zone_id`
LEFT JOIN `docks` d ON d.`dock_id` = ole.`dock_id`
LEFT JOIN `trailers` t ON t.`trailer_id` = ole.`trailer_id`
LEFT JOIN `trucks` tr ON tr.`truck_id` = ole.`truck_id`
WHERE ole.`box_id` IS NOT NULL;

CREATE OR REPLACE VIEW `vw_box_current_lifecycle` AS
SELECT history.*
FROM `vw_box_lifecycle_history` history
JOIN (
    SELECT `box_id`, MAX(`order_lifecycle_event_id`) AS `latest_event_id`
    FROM `order_lifecycle_events`
    WHERE `box_id` IS NOT NULL
    GROUP BY `box_id`
) latest
    ON latest.`latest_event_id` = history.`box_lifecycle_event_id`;

CREATE OR REPLACE VIEW `v_box_journey` AS
SELECT * FROM `vw_box_lifecycle_history`;

CREATE OR REPLACE VIEW `v_box_current_location` AS
SELECT
    current_box.`box_lifecycle_event_id`,
    current_box.`box_id`,
    current_box.`box_code`,
    current_box.`box_status`,
    current_box.`lifecycle_status`,
    current_box.`conveyor_code`,
    current_box.`pallet_code`,
    current_box.`zone_code`,
    current_box.`zone_name`,
    current_box.`dock_code`,
    current_box.`trailer_code`,
    current_box.`truck_number`,
    current_box.`event_time`
FROM `vw_box_current_lifecycle` current_box;

CREATE OR REPLACE VIEW `vw_conveyor_boxes` AS
SELECT *
FROM `vw_box_current_lifecycle`
WHERE `conveyor_code` IS NOT NULL
   OR UPPER(`lifecycle_status`) LIKE '%CONVEYOR%';

CREATE OR REPLACE VIEW `v_pallet_box_count` AS
SELECT
    p.`pallet_id`,
    p.`pallet_code`,
    p.`pallet_status`,
    COUNT(current_box.`box_id`) AS `total_boxes`,
    COALESCE(SUM(b.`weight_kg`), 0) AS `total_weight_kg`,
    COALESCE(SUM(b.`volume_cm3`), 0) AS `total_volume_cm3`
FROM `pallets` p
LEFT JOIN `vw_box_current_lifecycle` current_box
    ON current_box.`pallet_code` = p.`pallet_code`
LEFT JOIN `boxes` b ON b.`box_id` = current_box.`box_id`
GROUP BY p.`pallet_id`, p.`pallet_code`, p.`pallet_status`;

CREATE OR REPLACE VIEW `vw_pallet_trailer_routes` AS
SELECT
    ole.`pallet_id`,
    p.`pallet_code`,
    ole.`zone_id`,
    z.`zone_code`,
    z.`zone_name`,
    ole.`dock_id`,
    d.`dock_code`,
    ole.`trailer_id`,
    t.`trailer_code`,
    ole.`truck_id`,
    tr.`truck_number`,
    ole.`trailer_dock_assignment_id`,
    ole.`trailer_truck_assignment_id`,
    ole.`lifecycle_status`,
    ole.`event_time`
FROM `order_lifecycle_events` ole
JOIN `pallets` p ON p.`pallet_id` = ole.`pallet_id`
LEFT JOIN `zones` z ON z.`zone_id` = ole.`zone_id`
LEFT JOIN `docks` d ON d.`dock_id` = ole.`dock_id`
LEFT JOIN `trailers` t ON t.`trailer_id` = ole.`trailer_id`
LEFT JOIN `trucks` tr ON tr.`truck_id` = ole.`truck_id`
JOIN (
    SELECT `pallet_id`, MAX(`order_lifecycle_event_id`) AS `latest_event_id`
    FROM `order_lifecycle_events`
    WHERE `pallet_id` IS NOT NULL
    GROUP BY `pallet_id`
) latest ON latest.`latest_event_id` = ole.`order_lifecycle_event_id`;

CREATE OR REPLACE VIEW `v_latest_video_run` AS
SELECT *
FROM `video_processing_runs`
WHERE `run_id` = (SELECT MAX(`run_id`) FROM `video_processing_runs`);

CREATE OR REPLACE VIEW `v_latest_video_pallets` AS
SELECT p.*
FROM `video_final_pallets` p
JOIN `v_latest_video_run` r ON r.`run_id` = p.`run_id`;

CREATE OR REPLACE VIEW `v_latest_video_boxes` AS
SELECT b.*
FROM `video_final_boxes` b
JOIN `v_latest_video_run` r ON r.`run_id` = b.`run_id`;

CREATE OR REPLACE VIEW `v_latest_video_events` AS
SELECT e.*
FROM `video_events` e
JOIN `v_latest_video_run` r ON r.`run_id` = e.`run_id`;

-- =====================================================================
-- 9. FINAL VERIFICATION
-- =====================================================================

SELECT
    TABLE_TYPE,
    COUNT(*) AS object_count
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = DATABASE()
GROUP BY TABLE_TYPE;

SELECT
    TABLE_NAME,
    TABLE_TYPE
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = DATABASE()
ORDER BY TABLE_TYPE, TABLE_NAME;
