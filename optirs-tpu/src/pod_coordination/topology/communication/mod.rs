// Communication management for topology module
//
// This module handles communication patterns, interfaces, routing, and
// quality of service for TPU pod topology.

// Re-export communication types
#[derive(Debug, Clone, Default)]
pub struct CommunicationChannel;

#[derive(Debug, Clone, Default)]
pub struct CommunicationInterface;

#[derive(Debug, Clone, Default)]
pub struct CommunicationLink;

#[derive(Debug, Clone, Default)]
pub struct CommunicationPath;

#[derive(Debug, Clone, Default)]
pub struct CommunicationPattern;

#[derive(Debug, Clone, Default)]
pub struct CommunicationQoS;

#[derive(Debug, Clone, Default)]
pub struct CommunicationRouting;

#[derive(Debug, Clone, Default)]
pub struct CommunicationTopology;
